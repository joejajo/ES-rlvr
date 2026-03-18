import re

from reward.reward_utils import extract_answer, grade_answer_sympy, grade_answer_mathd


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def compute_training_score(solution_str: str, ground_truth: str) -> float:
    """Binary reward for training. Returns 1.0 (correct) or 0.0 (wrong)."""
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
    """One-shot RLVR binary reward: 1.0 if \\boxed{} answer matches GT, else 0.0."""
    if use_think is False:
        model_solution = solution_str
    elif solution_str and "<think>" in solution_str and "</think>" in solution_str:
        model_solution = solution_str.split("</think>", 1)[1]
    else:
        return 0.0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0.0

    if isinstance(ground_truth, (str, float, int)):
        ground_truths = [ground_truth]
    elif ground_truth is None:
        ground_truths = []
    else:
        ground_truths = list(ground_truth)

    processed = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            extracted = extract_answer(truth)
            if extracted is not None:
                processed.append(extracted)
        else:
            processed.append(truth)

    if not processed:
        return 0.0

    for gt in processed:
        if grade_answer_mathd(model_answer, gt) or grade_answer_sympy(model_answer, gt):
            return 1.0
    return 0.0
