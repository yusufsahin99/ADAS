'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import os
import sys
import torch
import re
import json
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
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


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    mode = os.environ.get("LLADA_SDP_MODE", "").strip().lower()
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

    raise ValueError(f"Unknown LLADA_SDP_MODE={mode!r}")


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

    # Keep percentage-style limits unchanged; only reinterpret integer-ish
    # test limits as a total budget across the group's subtasks.
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


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        sampler="baseline",
        remasking='low_confidence',
        attention_discount_alpha=40.0,
        attention_discount_mode="candidate_to_selected",
        attention_discount_score_rule="additive",
        attention_discount_normalization="raw",
        attention_discount_aggregation="sequential",
        attention_discount_selected_uncertainty="confidence_gap",
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
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        legacy_gamma = kwargs.pop("dbes_gamma", None)
        if legacy_gamma is not None:
            if float(gamma) != 1.0 and float(legacy_gamma) != float(gamma):
                raise ValueError("Pass only one of gamma or dbes_gamma in model_args.")
            gamma = legacy_gamma

        self.sdpa_mode = _configure_sdpa_from_env()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
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
        assert self.tokenizer.pad_token_id != self.mask_id

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.sampler = sampler
        self.remasking = remasking
        self.attention_discount_alpha = attention_discount_alpha
        self.attention_discount_mode = attention_discount_mode
        self.attention_discount_score_rule = attention_discount_score_rule
        self.attention_discount_normalization = attention_discount_normalization
        self.attention_discount_aggregation = attention_discount_aggregation
        self.attention_discount_selected_uncertainty = attention_discount_selected_uncertainty
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

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        entries = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]

        out = []
        stats_records = [] if self.generation_stats_path else None
        generation_model = self.model.module if hasattr(self.model, "module") else self.model
        recorded_sampler_variant = "confidence_topk" if self.sampler == "baseline" else self.sampler_variant
        for batch_start in tqdm(range(0, len(entries), self.generation_batch_size), desc="Generating..."):
            batch_entries = entries[batch_start : batch_start + self.generation_batch_size]
            encoded = self.tokenizer(
                [e["question"] for e in batch_entries],
                add_special_tokens=False,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            prompt = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
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
                temperature=0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
                attention_discount_alpha=self.attention_discount_alpha,
                attention_discount_mode=self.attention_discount_mode,
                attention_discount_score_rule=self.attention_discount_score_rule,
                attention_discount_normalization=self.attention_discount_normalization,
                attention_discount_aggregation=self.attention_discount_aggregation,
                attention_discount_selected_uncertainty=self.attention_discount_selected_uncertainty,
                attention_capture_aggregation=self.attention_capture_aggregation,
                attention_capture_last_k_layers=self.attention_capture_last_k_layers,
                attention_capture_backend=self.attention_capture_backend,
                sampler_variant=recorded_sampler_variant,
                gamma=self.gamma,
                abnn_f=self.abnn_f,
                pad_to_max_seq_len=self.pad_to_max_seq_len,
                max_seq_len=self.max_length,
                pad_token_id=self.tokenizer.pad_token_id or 0,
                return_stats=want_stats,
            )

            if want_stats:
                generated_answer, generation_stats = generated_result
            else:
                generated_answer = generated_result

            for sample_idx, elem in enumerate(batch_entries):
                generated_text = self.tokenizer.decode(generated_answer[sample_idx][prompt.shape[1]:], skip_special_tokens=False)
                for stop_seq in elem["until"]:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]

                generated_answer_ids = self.tokenizer(generated_text)["input_ids"]
                generated_text = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                out.append(generated_text)
                if stats_records is not None:
                    stats_records.append(
                        {
                            "request_index": batch_start + sample_idx,
                            "prompt_tokens": int(attention_mask[sample_idx].sum().item()),
                            "generated_tokens": int(len(generated_answer_ids)),
                            "nfe": int(generation_stats["nfe_per_sample"][sample_idx]),
                            "selected_counts": generation_stats["selected_counts_per_sample"][sample_idx],
                            "selected_risks": generation_stats.get("selected_risks_per_sample", [[] for _ in range(prompt.shape[0])])[sample_idx],
                            "step_budget_values": generation_stats.get("step_budget_values_per_sample", [[] for _ in range(prompt.shape[0])])[sample_idx],
                            "selection_overlap_ratio": generation_stats.get("selection_overlap_ratio_per_sample", [[] for _ in range(prompt.shape[0])])[sample_idx],
                            "selection_replacement_count": generation_stats.get("selection_replacement_count_per_sample", [[] for _ in range(prompt.shape[0])])[sample_idx],
                            "selected_base_score_mean": generation_stats.get("selected_base_score_mean_per_sample", [[] for _ in range(prompt.shape[0])])[sample_idx],
                            "selected_effective_score_mean": generation_stats.get("selected_effective_score_mean_per_sample", [[] for _ in range(prompt.shape[0])])[sample_idx],
                            "selected_score_shift_mean": generation_stats.get("selected_score_shift_mean_per_sample", [[] for _ in range(prompt.shape[0])])[sample_idx],
                        }
                    )

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

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
    
