import re

from utils.reward_utils import extract_answer, grade_answer_sympy, grade_answer_mathd


# System prompt — matches One-Shot-RLVR / qwen25-math-cot exactly.
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# data_sources routed to math.compute_score (Hendrycks / EleutherAI grader)
_MATH_SOURCES = {
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "simplerl/math500",
    "simplerl/math_level3to5",
    "simplerl/aime24",
    "Maxwell-Jia/AIME_2024",
    "gneubig/aime-1983-2024",
}

# data_sources routed to compute_score (deepscaler grader)
_DEEPSCALER_SOURCES = {
    "deepscaler", "aime", "amc", "math", "minerva", "olympiad",
}


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


def compute_score_routed(data_source: str, solution_str: str,
                         ground_truth: str, extra_info=None) -> float:
    """
    Data-source-aware routing — mirrors One-Shot-RLVR _default_compute_score.

    simplerl/math500 and related MATH sources → utils.math_score.compute_score
        (Hendrycks / EleutherAI strip_string + is_equiv grader)
    deepscaler / aime / amc / math / minerva / olympiad → compute_score
        (dual-pass grade_answer_mathd OR grade_answer_sympy grader)
    All others fall back to compute_score.
    """
    if data_source in _MATH_SOURCES:
        from utils.math_score import compute_score as math_compute_score
        return float(math_compute_score(solution_str, ground_truth))

    return float(compute_score(data_source, solution_str, ground_truth,
                               extra_info=extra_info, use_think=False))
