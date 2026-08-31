import ast
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from lm_eval.tasks.humaneval.utils import build_predictions, pass_at_k

MATH_PAPER_4SHOT_PREFIX = """
Problem:
Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.

Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$

Solution:
We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{align*}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations
\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}
has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$ Since we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.
""".strip()


def _repo_root():
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _import_math_helpers():
    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        right_brace_idx = None
        num_left_braces_open = 0
        for i in range(idx, len(string)):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break

        if right_brace_idx is None:
            return None
        return string[idx : right_brace_idx + 1]

    def remove_boxed(s):
        left = "\\boxed{"
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            return s[len(left) : -1]
        except Exception:
            return None

    def normalize_final_answer(final_answer: str) -> str:
        substitutions = [
            ("an ", ""),
            ("a ", ""),
            (".$", "$"),
            ("\\$", ""),
            (r"\ ", ""),
            (" ", ""),
            ("mbox", "text"),
            (",\\text{and}", ","),
            ("\\text{and}", ","),
            ("\\text{m}", "\\text{}"),
            ("\\le", "<"),
        ]
        removed_expressions = [
            "square",
            "ways",
            "integers",
            "dollars",
            "mph",
            "inches",
            "ft",
            "hours",
            "km",
            "units",
            "\\ldots",
            "sue",
            "points",
            "feet",
            "minutes",
            "digits",
            "cents",
            "degrees",
            "cm",
            "gm",
            "pounds",
            "meters",
            "meals",
            "edges",
            "students",
            "childrentickets",
            "multiples",
            "\\text{s}",
            "\\text{.}",
            "\\text{\ns}",
            "\\text{}^2",
            "\\text{}^3",
            "\\text{\n}",
            "\\text{}",
            r"\mathrm{th}",
            r"^\circ",
            r"^{\circ}",
            r"\;",
            r",\!",
            "{,}",
            '"',
            "\\dots",
            "\n",
            "\r",
            "\f",
        ]
        for before, after in substitutions:
            final_answer = final_answer.replace(before, after)
        for expr in removed_expressions:
            final_answer = final_answer.replace(expr, "")

        final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
        if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
            final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]
        if len(re.findall(r"answer?is:?(.*)", final_answer)) > 0:
            final_answer = re.findall(r"answer?is:?(.*)", final_answer)[-1]
        if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
            final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]
        if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
            final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]

        final_answer = final_answer.strip()
        if "rac" in final_answer and "\\frac" not in final_answer:
            final_answer = final_answer.replace("rac", "\\frac")
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")
        return final_answer

    def math_postprocess(text: str) -> str:
        for maybe_ans in text.split("."):
            if "final answer" in maybe_ans.lower():
                return normalize_final_answer(maybe_ans)
        return normalize_final_answer(text.split(".")[0])

    class MATHEvaluator:
        def _fix_fracs(self, string):
            substrs = string.split("\\frac")
            new_str = substrs[0]
            if len(substrs) > 1:
                for substr in substrs[1:]:
                    new_str += "\\frac"
                    if len(substr) > 0 and substr[0] == "{":
                        new_str += substr
                    else:
                        try:
                            assert len(substr) >= 2
                        except AssertionError:
                            return string
                        a = substr[0]
                        b = substr[1]
                        if b != "{":
                            post_substr = substr[2:] if len(substr) > 2 else ""
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            post_substr = substr[2:] if len(substr) > 2 else ""
                            new_str += "{" + a + "}" + b + post_substr
            return new_str

        def _fix_a_slash_b(self, string):
            if len(string.split("/")) != 2:
                return string
            a, b = string.split("/")
            try:
                a = int(a)
                b = int(b)
                assert string == f"{a}/{b}"
                return "\\frac{" + str(a) + "}{" + str(b) + "}"
            except Exception:
                return string

        def _remove_right_units(self, string):
            if "\\text{ " in string:
                splits = string.split("\\text{ ")
                assert len(splits) == 2
                return splits[0]
            return string

        def _fix_sqrt(self, string):
            if "\\sqrt" not in string:
                return string
            splits = string.split("\\sqrt")
            new_string = splits[0]
            for split in splits[1:]:
                if split and split[0] != "{":
                    new_string += "\\sqrt{" + split[0] + "}" + split[1:]
                else:
                    new_string += "\\sqrt" + split
            return new_string

        def _strip_string(self, string):
            string = string.replace("\n", "")
            string = string.replace("\\!", "")
            string = string.replace("\\\\", "\\")
            string = string.replace("tfrac", "frac").replace("dfrac", "frac")
            string = string.replace("\\left", "").replace("\\right", "")
            string = string.replace("^{\\circ}", "").replace("^\\circ", "")
            string = string.replace("\\$", "")
            string = self._remove_right_units(string)
            string = string.replace("\\%", "").replace("%", "")
            string = string.replace(" .", " 0.").replace("{.", "{0.")
            if len(string) == 0:
                return string
            if string[0] == ".":
                string = "0" + string
            if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
                string = string.split("=")[1]
            string = self._fix_sqrt(string)
            string = string.replace(" ", "")
            string = self._fix_fracs(string)
            if string == "0.5":
                string = "\\frac{1}{2}"
            return self._fix_a_slash_b(string)

        def is_equiv(self, str1, str2, verbose=False):
            if str1 is None and str2 is None:
                return True
            if str1 is None or str2 is None:
                return False
            try:
                ss1 = self._strip_string(str1)
                ss2 = self._strip_string(str2)
                if verbose:
                    print(ss1, ss2)
                if ss1 == ss2:
                    return True
                if normalize_final_answer(ss1) == normalize_final_answer(ss2):
                    return True
            except Exception:
                pass
            try:
                if normalize_final_answer(str1) == normalize_final_answer(str2):
                    return True
            except Exception:
                pass
            return str1 == str2

    return {
        "math_postprocess": math_postprocess,
        "normalize_final_answer": normalize_final_answer,
        "remove_boxed": remove_boxed,
        "last_boxed_only_string": last_boxed_only_string,
        "evaluator": MATHEvaluator(),
    }


def hendrycks_math_paper_doc_to_text(doc):
    return f"{MATH_PAPER_4SHOT_PREFIX}\n\nProblem:\n{doc['problem']}\nSolution:\n"


def hendrycks_math_paper_process_docs(dataset):
    helpers = _import_math_helpers()
    normalize_final_answer = helpers["normalize_final_answer"]
    remove_boxed = helpers["remove_boxed"]
    last_boxed_only_string = helpers["last_boxed_only_string"]

    def _process_doc(doc):
        answer = normalize_final_answer(
            remove_boxed(last_boxed_only_string(doc["solution"]))
        )
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": answer,
        }

    return dataset.map(_process_doc)


def hendrycks_math_paper_process_results(doc, results):
    helpers = _import_math_helpers()
    prediction = helpers["math_postprocess"](results[0])
    exact = 1 if prediction and helpers["evaluator"].is_equiv(prediction, doc["answer"]) else 0
    return {"exact_match": exact}


def _dream_humaneval_refine_text(text):
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def _dream_humaneval_syntax_check(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False


def _dream_humaneval_extract_longest_valid_code(text):
    lines = text.splitlines()
    if len(lines) > 100:
        lines = lines[:100]

    max_valid_lines = 0
    max_valid_snippet = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i : j + 1])
            if not _dream_humaneval_syntax_check(current_snippet):
                continue
            valid_line_count = sum(1 for line in lines[i : j + 1] if line.strip())
            if valid_line_count > max_valid_lines:
                max_valid_lines = valid_line_count
                max_valid_snippet = current_snippet
    return max_valid_snippet


def _dream_humaneval_get_deps(nodes):
    name_to_deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    deps.add(child.attr)
                else:
                    stack.append(child)
        name_to_deps[name] = deps
    return name_to_deps


def _dream_humaneval_get_function_dependency(entrypoint, call_graph):
    visited = set()
    to_visit = [entrypoint]

    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)

    return visited


def _dream_humaneval_get_definition_name(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def _dream_humaneval_has_return_statement(node):
    return any(isinstance(n, ast.Return) for n in ast.walk(node))


def _dream_humaneval_sanitize(text, entrypoint=None):
    text = _dream_humaneval_refine_text(text)
    code = _dream_humaneval_extract_longest_valid_code(text)
    if not code.strip():
        return ""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    definitions = {}
    imports = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = ("class", node)
        elif isinstance(node, ast.FunctionDef):
            if _dream_humaneval_has_return_statement(node):
                definitions[node.name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = _dream_humaneval_get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)

    reachable = set(definitions)
    if entrypoint:
        name_to_deps = _dream_humaneval_get_deps(
            [(name, node) for name, (_, node) in definitions.items()]
        )
        reachable = _dream_humaneval_get_function_dependency(entrypoint, name_to_deps)

    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))
    for name, (_, node) in definitions.items():
        if not entrypoint or name in reachable:
            sanitized_output.append(ast.unparse(node))
    return "\n".join(sanitized_output)


def _dream_humaneval_extract_candidate(text):
    if not isinstance(text, str):
        text = str(text)

    try:
        evaluated = ast.literal_eval(text)
    except Exception:
        pass
    else:
        if isinstance(evaluated, str):
            text = evaluated

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if text.startswith("Here"):
        text = "\n".join(text.split("\n")[1:]).strip("\n")

    candidate = text.split("```python\n", 1)[-1]
    candidate = candidate.split("```", 1)[0]

    if candidate.lstrip().startswith("python\n"):
        stripped = candidate.lstrip()
        remainder = stripped[len("python\n") :]
        if remainder.lstrip().startswith(("def ", "class ", "from ", "import ", "@")):
            candidate = remainder

    return candidate.strip("\n")


def build_predictions_dream_humaneval(resps, docs):
    predictions = []
    for resp, doc in zip(resps, docs):
        row_predictions = []
        for raw_resp in resp:
            candidate = _dream_humaneval_extract_candidate(raw_resp)
            full_text = f"{doc['prompt']}{candidate}" if candidate else doc["prompt"]
            sanitized = _dream_humaneval_sanitize(full_text, doc.get("entry_point"))
            row_predictions.append(sanitized or full_text)
        predictions.append(row_predictions)
    return predictions


__all__ = [
    "build_predictions",
    "build_predictions_dream_humaneval",
    "pass_at_k",
    "hendrycks_math_paper_doc_to_text",
    "hendrycks_math_paper_process_docs",
    "hendrycks_math_paper_process_results",
]


MINERVA_SUBTASK_SIZES = {
    "algebra": 1187,
    "counting_and_probability": 474,
    "geometry": 479,
    "intermediate_algebra": 903,
    "number_theory": 540,
    "prealgebra": 871,
    "precalculus": 546,
}


@lru_cache(maxsize=1)
def _import_minerva_math_helpers():
    from lm_eval.tasks.minerva_math import utils as minerva_utils  # noqa: WPS433

    return minerva_utils


def _minerva_total_limit():
    return int(os.environ.get("MINERVA_TOTAL_LIMIT", "100"))


def _minerva_shuffle_seed():
    return int(os.environ.get("MINERVA_SAMPLE_SEED", "1234"))


def _minerva_quota_map(total_limit: int):
    total_size = sum(MINERVA_SUBTASK_SIZES.values())
    exact = {
        key: (size * total_limit) / total_size for key, size in MINERVA_SUBTASK_SIZES.items()
    }
    quota = {key: int(value) for key, value in exact.items()}
    remaining = total_limit - sum(quota.values())
    for key, _ in sorted(
        ((key, exact[key] - quota[key]) for key in quota),
        key=lambda item: item[1],
        reverse=True,
    )[:remaining]:
        quota[key] += 1
    return quota


def _minerva_process_docs_for_subtask(dataset, subtask_key: str):
    minerva_utils = _import_minerva_math_helpers()
    processed = minerva_utils.process_docs(dataset)
    shuffled = processed.shuffle(seed=_minerva_shuffle_seed())
    quota = _minerva_quota_map(_minerva_total_limit())[subtask_key]
    quota = min(quota, len(shuffled))
    return shuffled.select(range(quota))


def minerva_math_doc_to_text(doc):
    return _import_minerva_math_helpers().doc_to_text(doc)


def minerva_math_process_results(doc, results):
    return _import_minerva_math_helpers().process_results(doc, results)


def minerva_math_list_fewshot_samples():
    return _import_minerva_math_helpers().list_fewshot_samples()


def minerva_math_process_docs_algebra(dataset):
    return _minerva_process_docs_for_subtask(dataset, "algebra")


def minerva_math_process_docs_counting_and_probability(dataset):
    return _minerva_process_docs_for_subtask(dataset, "counting_and_probability")


def minerva_math_process_docs_geometry(dataset):
    return _minerva_process_docs_for_subtask(dataset, "geometry")


def minerva_math_process_docs_intermediate_algebra(dataset):
    return _minerva_process_docs_for_subtask(dataset, "intermediate_algebra")


def minerva_math_process_docs_number_theory(dataset):
    return _minerva_process_docs_for_subtask(dataset, "number_theory")


def minerva_math_process_docs_prealgebra(dataset):
    return _minerva_process_docs_for_subtask(dataset, "prealgebra")


def minerva_math_process_docs_precalculus(dataset):
    return _minerva_process_docs_for_subtask(dataset, "precalculus")


__all__.extend(
    [
        "minerva_math_doc_to_text",
        "minerva_math_process_results",
        "minerva_math_list_fewshot_samples",
        "minerva_math_process_docs_algebra",
        "minerva_math_process_docs_counting_and_probability",
        "minerva_math_process_docs_geometry",
        "minerva_math_process_docs_intermediate_algebra",
        "minerva_math_process_docs_number_theory",
        "minerva_math_process_docs_prealgebra",
        "minerva_math_process_docs_precalculus",
    ]
)
