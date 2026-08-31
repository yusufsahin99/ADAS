import accelerate
import ast
import os
import random
import re
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from attention_guided_sampling import generate_attention_discounted

GROUP_TOTAL_LIMIT_SUBTASKS = {
    "hendrycks_math": 7,
    "hendrycks_math_paper": 7,
}

TASK_ALIASES = {
    "hendrycks_math_paper": [
        "hendrycks_math_paper_algebra",
        "hendrycks_math_paper_counting_and_prob",
        "hendrycks_math_paper_geometry",
        "hendrycks_math_paper_intermediate_algebra",
        "hendrycks_math_paper_num_theory",
        "hendrycks_math_paper_prealgebra",
        "hendrycks_math_paper_precalc",
    ],
}


_CODE_TASK_PREFIXES = ("humaneval", "mbpp")
_CODE_LINE_START_RE = re.compile(r"(?m)^(?:from\b|import\b|def\b|class\b|@)")
_FENCED_CODE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _configure_sdpa_from_env():
    mode = os.environ.get("DREAM_SDP_MODE", "").strip().lower()
    if not mode:
        return "default"

    if not hasattr(torch.backends, "cuda"):
        return mode

    if mode == "default":
        return mode
    if mode == "no_flash":
        torch.backends.cuda.enable_flash_sdp(False)
        return mode
    if mode == "math_only":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return mode
    if mode == "flash_only":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        return mode

    raise ValueError(f"Unknown DREAM_SDP_MODE={mode!r}")


def _get_cli_option(argv, name):
    for idx, arg in enumerate(argv):
        if arg == name and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _set_cli_option(argv, name, value):
    for idx, arg in enumerate(argv):
        if arg == name and idx + 1 < len(argv):
            argv[idx + 1] = value
            return True
        if arg.startswith(name + "="):
            argv[idx] = f"{name}={value}"
            return True
    return False


def _maybe_adjust_group_total_limit(argv):
    if os.environ.get("DISABLE_GROUP_TOTAL_LIMIT_REWRITE", "").lower() in {"1", "true", "yes"}:
        return
    tasks_value = _get_cli_option(argv, "--tasks")
    limit_value = _get_cli_option(argv, "--limit")
    if not tasks_value or not limit_value:
        return

    task_names = [task.strip() for task in tasks_value.split(",") if task.strip()]
    if len(task_names) != 1:
        return

    task_name = task_names[0]
    subtask_count = GROUP_TOTAL_LIMIT_SUBTASKS.get(task_name)
    if subtask_count is None:
        return

    try:
        parsed_limit = float(limit_value)
    except ValueError:
        return

    if parsed_limit < 1:
        return

    total_limit = int(parsed_limit)
    per_subtask_limit = max(1, total_limit // subtask_count)
    if per_subtask_limit == total_limit:
        return

    if _set_cli_option(argv, "--limit", str(per_subtask_limit)):
        print(
            f"[limit-adjust] task={task_name} total_limit={total_limit} "
            f"subtasks={subtask_count} -> per_subtask_limit={per_subtask_limit} "
            f"(approx total={per_subtask_limit * subtask_count})"
        )


def _maybe_expand_task_aliases(argv):
    tasks_value = _get_cli_option(argv, "--tasks")
    if not tasks_value:
        return

    task_names = [task.strip() for task in tasks_value.split(",") if task.strip()]
    expanded = []
    changed = False
    for task_name in task_names:
        alias_targets = TASK_ALIASES.get(task_name)
        if alias_targets is None:
            expanded.append(task_name)
            continue
        expanded.extend(alias_targets)
        changed = True

    if changed and _set_cli_option(argv, "--tasks", ",".join(expanded)):
        print(f"[task-expand] tasks={tasks_value} -> {','.join(expanded)}")


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_code_task(task_name):
    return isinstance(task_name, str) and task_name.startswith(_CODE_TASK_PREFIXES)


def _format_dream_prompt(tokenizer, prompt_text, apply_chat_template, task_name):
    if not apply_chat_template:
        return prompt_text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _dedupe_candidates(candidates):
    seen = set()
    deduped = []
    for candidate in candidates:
        candidate = candidate.strip("\n")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _code_candidates(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    candidates = [text]

    if "[DONE]" in text:
        candidates.append(text.split("[DONE]", 1)[0].strip())

    for match in _FENCED_CODE_RE.finditer(text):
        candidates.append(match.group(1).strip())

    unfenced = re.sub(r"```(?:python)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    candidates.append(unfenced)

    code_match = _CODE_LINE_START_RE.search(unfenced)
    if code_match:
        candidates.append(unfenced[code_match.start() :].strip())

    return _dedupe_candidates(candidates)


def _longest_parsable_prefix(text, prefix=""):
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not candidate:
            continue
        try:
            ast.parse(prefix + candidate)
            return candidate
        except SyntaxError:
            continue
    return None


def _extract_first_function_body(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            start = node.body[0].lineno - 1
            end = getattr(node.body[-1], "end_lineno", None)
            if end is None:
                end = len(lines)
            body = "\n".join(lines[start:end]).rstrip()
            return body or None
    return None


def _postprocess_humaneval_output(text, prompt_text):
    text = text.lstrip("\n")
    if text.startswith("Here"):
        text = "\n".join(text.split("\n")[1:]).strip()

    if "```" in text:
        blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split("```", 1)[-1]
        else:
            text = blocks[0]
            if not text.startswith("\n"):
                text = text[max(text.find("\n") + 1, 0) :]

    if text.strip().startswith(("from", "import")):
        def_idx = text.find("def")
        if def_idx != -1:
            text = text[max(text.find("\n", def_idx) + 1, 0) :]

    if text.strip().startswith("def"):
        text = "\n".join(text.split("\n")[1:])

    prediction = text.split("\n\n\n")[0]
    prediction = prediction.split("\n```")[0]

    # Match the OpenCompass Humaneval postprocess more closely, since the EB
    # paper notes Dream often emits code blocks that need to be unwrapped.
    oc_text = prediction
    oc_text = "\n".join([line for line in oc_text.split("\n") if line != ""])
    oc_text = oc_text.lstrip("\n")
    if oc_text.strip().startswith("def"):
        oc_text = "\n".join(oc_text.split("\n")[1:])
    if oc_text.startswith(" "):
        oc_text = "    " + oc_text.lstrip()
    else:
        oc_text = "\n".join(("    " + line) if line else line for line in oc_text.split("\n"))
    oc_lines = oc_text.split("\n")

    min_leading_space = None
    end_index = None
    for index, line in enumerate(oc_lines):
        stripped = line.strip()
        if not stripped or stripped[0] in ["'", '"', "#"]:
            continue
        current_leading_space = len(line.rstrip()) - len(stripped)
        if min_leading_space is None:
            min_leading_space = current_leading_space
        elif current_leading_space < min_leading_space:
            end_index = index
            break
    if end_index is not None:
        oc_text = "\n".join(oc_lines[:end_index])
    else:
        oc_text = "\n".join(oc_lines)
    oc_text = oc_text.rstrip()

    if oc_text:
        return oc_text + "\n"

    for candidate in _code_candidates(prediction):
        body = _extract_first_function_body(candidate)
        if body:
            return body.rstrip() + "\n"

    return ""


def _is_stock_humaneval_until(stop_tokens):
    return stop_tokens == ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]


def _postprocess_mbpp_output(text):
    patterns = [
        r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
        r"BEGIN\s*'(.*)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
        r"BEGIN\s*'(.*)\s*\[DONE\]",
        r"\[BEGIN\]\s*(.*)\s*\[DONE\]",
        r"BEGIN\s*(.*)\s*\[DONE\]",
        r"```python\s*(.*)\s*```",
        r"```\s*(.*)\s*```",
        r"```python\s*(.*)\s*$",
        r"```\s*(.*)\s*$",
        r"(.*)\s*```.*",
        r"\[BEGIN\]\s*'(.*)",
        r"\[BEGIN\](.*)",
        r"'(.*)'\s*\[DONE\]",
    ]

    text = text.strip()
    if text.startswith("Here"):
        text = "\n".join(text.split("\n")[1:]).strip()

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            break

    text = text.split("```")[0]
    text = re.split(r"'?\s*\[?DONE\]?", text)[0]
    text = text.replace("\\_", "_").strip()
    if text.startswith("'"):
        text = text[1:]
    if text.endswith("'"):
        text = text[:-1]
    return text.rstrip() + ("\n" if text else "")


def _postprocess_generation(task_name, text, prompt_text):
    if not isinstance(task_name, str):
        return text
    if task_name.startswith("humaneval"):
        return _postprocess_humaneval_output(text, prompt_text)
    if task_name.startswith("mbpp"):
        return _postprocess_mbpp_output(text)
    return text


@register_model("dream_dist")
class DreamEvalHarness(LM):
    def __init__(
        self,
        model_path="Dream-org/Dream-v0-Instruct-7B",
        max_length=2048,
        gen_length=256,
        block_length=None,
        steps=32,
        sampler="baseline",
        temperature=0.1,
        top_p=0.9,
        top_k=None,
        batch_size=1,
        mask_id=None,
        apply_chat_template=True,
        attention_discount_alpha=40.0,
        attention_discount_mode="candidate_to_selected",
        attention_discount_score_rule="additive",
        attention_discount_normalization="raw",
        attention_discount_aggregation="sequential",
        attention_discount_selected_uncertainty="confidence_gap",
        attention_discount_base_alg="maskgit_plus",
        attention_capture_aggregation="last_layer_mean",
        attention_capture_last_k_layers=2,
        attention_capture_backend="auto",
        sampler_variant="attention_discounted",
        gamma=1.0,
        abnn_f=1.0,
        pad_to_max_seq_len=False,
        generation_batch_size=1,
        generation_stats_path=None,
        device="cuda",
        **kwargs,
    ):
        super().__init__()

        self.sdpa_mode = _configure_sdpa_from_env()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        legacy_gamma = kwargs.pop("dbes_gamma", None)
        if legacy_gamma is not None:
            if float(gamma) != 1.0 and float(legacy_gamma) != float(gamma):
                raise ValueError("Pass only one of gamma or dbes_gamma in model_args.")
            gamma = legacy_gamma

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                raise ValueError("Tokenizer needs a pad token for batched generation.")

        self.max_length = int(max_length)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length or gen_length)
        self.steps = int(steps)
        self.sampler = sampler
        self.temperature = float(temperature)
        self.top_p = None if top_p in (None, "None") else float(top_p)
        self.top_k = None if top_k in (None, "None") else int(top_k)
        self.batch_size = int(batch_size)
        self.apply_chat_template = _parse_bool(apply_chat_template)

        resolved_mask_id = self.tokenizer.mask_token_id if mask_id is None else int(mask_id)
        if resolved_mask_id is None:
            raise ValueError("Dream tokenizer does not expose a mask token id; pass mask_id explicitly.")
        self.mask_id = resolved_mask_id

        self.attention_discount_alpha = attention_discount_alpha
        self.attention_discount_mode = attention_discount_mode
        self.attention_discount_score_rule = attention_discount_score_rule
        self.attention_discount_normalization = attention_discount_normalization
        self.attention_discount_aggregation = attention_discount_aggregation
        self.attention_discount_selected_uncertainty = attention_discount_selected_uncertainty
        self.attention_discount_base_alg = attention_discount_base_alg
        self.attention_capture_aggregation = attention_capture_aggregation
        self.attention_capture_last_k_layers = attention_capture_last_k_layers
        self.attention_capture_backend = attention_capture_backend
        self.sampler_variant = sampler_variant
        self.gamma = float(gamma)
        self.abnn_f = float(abnn_f)
        self.pad_to_max_seq_len = _parse_bool(pad_to_max_seq_len)
        self.generation_batch_size = int(generation_batch_size)
        self.generation_stats_path = generation_stats_path

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError("dream_dist currently supports generation-only tasks.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("dream_dist currently supports generation-only tasks.")

    def _format_prompt(self, prompt_text, task_name=None):
        return _format_dream_prompt(
            self.tokenizer,
            prompt_text,
            self.apply_chat_template,
            task_name,
        )

    def generate_until(self, requests: list[Instance]):
        generation_model = self.model.module if hasattr(self.model, "module") else self.model
        if self.pad_to_max_seq_len:
            if self.max_length < self.gen_length + 1:
                raise ValueError(
                    f"max_length={self.max_length} is too small for gen_length={self.gen_length}."
                )
            max_prompt_len = self.max_length - self.gen_length
        else:
            max_prompt_len = None
        ds = [
            {
                "question": req.args[0],
                "until": req.args[1]["until"],
                "task_name": getattr(req, "task_name", None),
            }
            for req in requests
        ]

        entries = []
        for e in ds:
            prompt_text = self._format_prompt(e["question"], e.get("task_name"))
            tokenized = self.tokenizer(prompt_text, add_special_tokens=False)
            input_ids = tokenized["input_ids"]
            if max_prompt_len is not None:
                input_ids = input_ids[-max_prompt_len:]
            else:
                input_ids = input_ids[-self.max_length :]
            entries.append(
                {
                    "input_ids": input_ids,
                    "question_text": prompt_text,
                    "until": e["until"],
                    "task_name": e.get("task_name"),
                }
            )

        out = []
        stats_records = [] if self.generation_stats_path else None
        recorded_sampler_variant = "confidence_topk" if self.sampler == "baseline" else self.sampler_variant
        for batch_start in tqdm(range(0, len(entries), self.generation_batch_size), desc="Generating..."):
            batch_entries = entries[batch_start : batch_start + self.generation_batch_size]
            pad_kwargs = {
                "return_attention_mask": True,
                "return_tensors": "pt",
            }
            if max_prompt_len is not None:
                pad_kwargs.update({"padding": "max_length", "max_length": max_prompt_len})
            else:
                pad_kwargs.update({"padding": True})
            padded = self.tokenizer.pad(
                [{"input_ids": e["input_ids"]} for e in batch_entries],
                **pad_kwargs,
            )
            prompt = padded["input_ids"].to(self.device)
            attention_mask = padded["attention_mask"].to(self.device)
            want_stats = stats_records is not None
            if self.sampler not in {"baseline", "attention_discounted"}:
                raise ValueError(f"Unknown sampler: {self.sampler}")

            generated_result = generate_attention_discounted(
                generation_model,
                prompt,
                attention_mask=attention_mask,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                remasking="low_confidence",
                mask_id=self.mask_id,
                attention_discount_alpha=self.attention_discount_alpha,
                attention_discount_mode=self.attention_discount_mode,
                attention_discount_score_rule=self.attention_discount_score_rule,
                attention_discount_normalization=self.attention_discount_normalization,
                attention_discount_aggregation=self.attention_discount_aggregation,
                attention_discount_selected_uncertainty=self.attention_discount_selected_uncertainty,
                attention_discount_base_alg=self.attention_discount_base_alg,
                attention_capture_aggregation=self.attention_capture_aggregation,
                attention_capture_last_k_layers=self.attention_capture_last_k_layers,
                attention_capture_backend=self.attention_capture_backend,
                sampler_variant=recorded_sampler_variant,
                gamma=self.gamma,
                abnn_f=self.abnn_f,
                return_stats=want_stats,
            )
            if want_stats:
                generated, generation_stats = generated_result
            else:
                generated = generated_result

            for row_idx, elem in enumerate(batch_entries):
                stop_tokens = elem["until"]
                prompt_text = elem["question_text"]
                task_name = elem["task_name"]

                generated_answer = self.tokenizer.decode(generated[row_idx][prompt.shape[1] :], skip_special_tokens=False)
                apply_stop_tokens = not (
                    task_name == "humaneval" and _is_stock_humaneval_until(stop_tokens)
                )
                if apply_stop_tokens:
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer:
                            generated_answer = generated_answer.split(stop_seq)[0]

                generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                generated_answer = _postprocess_generation(task_name, generated_answer, prompt_text)
                out.append(generated_answer)
                if stats_records is not None:
                    stats_records.append(
                        {
                            "request_index": len(out) - 1,
                            "prompt_tokens": int(attention_mask[row_idx].sum().item()),
                            "generated_tokens": int(len(generated_answer_ids)),
                            "nfe": int(generation_stats["nfe_per_sample"][row_idx]),
                            "selected_counts": generation_stats["selected_counts_per_sample"][row_idx],
                            "selected_risks": generation_stats.get("selected_risks_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "step_budget_values": generation_stats.get("step_budget_values_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selected_confidences": generation_stats.get("selected_confidences_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selected_uncertainty_weights": generation_stats.get("selected_uncertainty_weights_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selected_raw_dependencies": generation_stats.get("selected_raw_dependencies_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selection_overlap_ratio": generation_stats.get("selection_overlap_ratio_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selection_replacement_count": generation_stats.get("selection_replacement_count_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selected_base_score_mean": generation_stats.get("selected_base_score_mean_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selected_effective_score_mean": generation_stats.get("selected_effective_score_mean_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                            "selected_score_shift_mean": generation_stats.get("selected_score_shift_mean_per_sample", [[] for _ in range(prompt.shape[0])])[row_idx],
                        }
                    )

        if stats_records is not None:
            stats_path = Path(self.generation_stats_path)
            if self.accelerator is not None and getattr(self.accelerator, "num_processes", 1) > 1:
                stats_path = stats_path.with_name(
                    f"{stats_path.stem}.rank{self.accelerator.process_index}{stats_path.suffix}"
                )
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_path.open("w") as f:
                json.dump(
                    {
                        "sampler": self.sampler,
                        "sampler_variant": recorded_sampler_variant,
                        "steps": int(self.steps),
                        "gen_length": int(self.gen_length),
                        "records": stats_records,
                    },
                    f,
                    indent=2,
                )
        return out


if __name__ == "__main__":
    set_seed(1234)
    _maybe_adjust_group_total_limit(sys.argv)
    _maybe_expand_task_aliases(sys.argv)
    cli_evaluate()
