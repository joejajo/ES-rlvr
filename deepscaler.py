import re

try:
    from verl.utils.reward_score.utils import extract_answer, grade_answer_sympy, grade_answer_mathd  # pyright: ignore[reportMissingImports]
except ImportError:
    try:
        from verl.utils.reward_score.utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd  # pyright: ignore[reportMissingImports]
    except ImportError:
        def extract_answer(text):
            if text is None:
                return None
            s = str(text)
            # Use the LAST \boxed{} (rfind), matching One-Shot-RLVR behaviour.
            idx = s.rfind(r"\boxed")
            if idx == -1:
                return None
            # Advance past \boxed and optional whitespace to the opening brace.
            i = idx + len(r"\boxed")
            while i < len(s) and s[i] == " ":
                i += 1
            if i >= len(s) or s[i] != "{":
                return None
            depth = 0
            start = i + 1  # content starts after the opening brace
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:j].strip()
            return None

        def grade_answer_sympy(model_answer, ground_truth):
            return str(model_answer).strip() == str(ground_truth).strip()

        def grade_answer_mathd(model_answer, ground_truth):
            return str(model_answer).strip() == str(ground_truth).strip()


# System prompt — matches One-Shot-RLVR / qwen25-math-cot exactly.
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def compute_training_score(solution_str: str, ground_truth: str) -> float:
    """
    Training reward. Identical grading logic to compute_score:
      1. Model output must contain \\boxed{<answer>} (last occurrence used).
      2. Ground truth that contains \\boxed{} is unwrapped the same way.
      3. Correct iff grade_answer_mathd OR grade_answer_sympy.

    Returns 1.0 (correct) or 0.0 (wrong).
    """
    model_answer = extract_answer(solution_str)
    if model_answer is None:
        return 0.0

    gt = str(ground_truth)
    if "\\boxed" in gt:
        gt = extract_answer(gt)
        if gt is None:
            return 0.0

    if grade_answer_mathd(model_answer, gt) or grade_answer_sympy(model_answer, gt):
        return 1.0
    return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, use_think=False):
    """
    One-shot RLVR style binary reward:
      1.0 -> extracted \boxed{} answer matches ground truth
      0.0 -> otherwise
    """
    # Step 1: handle optional <think>...</think> wrappers.
    if use_think is False:
        model_solution = solution_str
    elif solution_str and "<think>" in solution_str and "</think>" in solution_str:
        model_solution = solution_str.split("</think>", 1)[1]
    else:
        return 0.0

    # Step 2: model must provide a boxed answer.
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0.0

    # Step 3: normalize ground truth candidates.
    if isinstance(ground_truth, (str, float, int)):
        ground_truths = [ground_truth]
    elif ground_truth is None:
        ground_truths = []
    else:
        ground_truths = list(ground_truth)

    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            extracted = extract_answer(truth)
            if extracted is not None:
                processed_ground_truths.append(extracted)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0.0

    # Step 4: strict correctness check.
    for gt in processed_ground_truths:
        if grade_answer_mathd(model_answer, gt) or grade_answer_sympy(model_answer, gt):
            return 1.0

    return 0.0
