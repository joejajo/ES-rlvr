"""
Math500 evaluator — single source of truth for validation grading.

Used by evaluate_val_set() in es_rlvr_train.py.

Pipeline (matches One-Shot-RLVR / Qwen2.5-Eval exactly):
  1. os_extract_answer : extract answer from model output
       - handles \\boxed{} with brace-counting
       - handles "final answer is $...$", "the answer is ..."
       - last-number fallback
  2. strip_string       : normalise ground truth LaTeX
  3. math_equal         : compare prediction vs ground truth
       - numeric (isclose tol=1e-4)
       - symbolic (SymPy / latex2sympy2)
       - matrix / tuple

Returns 1.0 (correct) or 0.0 (wrong).
"""

from utils.os_parser import extract_answer as os_extract_answer, strip_string
from utils.os_grader import math_equal


def grade_answer(model_output: str, ground_truth: str) -> float:
    """
    Grade one math500 model output against its ground truth.

    Parameters
    ----------
    model_output  : raw text the model generated (full CoT + answer)
    ground_truth  : ground truth string from the dataset

    Returns
    -------
    1.0 if correct, 0.0 otherwise.
    """
    pred = os_extract_answer(model_output, data_name="math500")
    if not pred:
        return 0.0
    gt = strip_string(str(ground_truth))
    return 1.0 if math_equal(pred, gt, timeout=False) else 0.0
