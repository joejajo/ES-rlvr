"""
eval/math500_grader.py — Independent MATH-500 answer extractor and grader.

This module is intentionally decoupled from the training reward pipeline.
``compute_training_score`` is never imported or called here.  The grading
helpers from ``reward.reward_utils`` (``grade_answer_mathd``,
``grade_answer_sympy``, ``last_boxed_only_string``) are reused read-only
so that the mathematical comparison logic stays in one place.

Two extraction modes
--------------------
strict (official / thesis number)
    Prefer the last ``\\boxed{...}``.  If absent, fall back to the last
    ``\\fbox{...}``.  If neither is present, return ``None`` → score 0.

relaxed (diagnostic / ablation only)
    First try strict extraction.  If that fails, try plain-text patterns:
        1. ``#### <answer>``   (GSM8K style)
        2. ``Final answer: <answer>``
        3. ``Answer: <answer>``
    The relaxed score must never be reported as the headline benchmark.
"""

from __future__ import annotations

import re
from typing import Optional

from reward.reward_utils import (
    grade_answer_mathd,
    grade_answer_sympy,
    last_boxed_only_string,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap_box(boxed_str: Optional[str]) -> Optional[str]:
    """Strip the outer ``\\boxed{...}`` or ``\\fbox{...}`` wrapper.

    Handles optional whitespace between the command and its opening brace,
    e.g. ``\\fbox {42}`` as well as the standard ``\\fbox{42}``.
    For nested content such as ``\\boxed{\\frac{1}{2}}``, only the outermost
    ``\\boxed{`` and closing ``}`` are stripped — inner braces are preserved
    correctly because ``last_boxed_only_string`` already matched the full
    brace-balanced span.

    Parameters
    ----------
    boxed_str:
        A string of the form ``\\boxed{content}`` or ``\\fbox{content}`` (with
        optional whitespace before ``{``) as returned by
        ``last_boxed_only_string``, or ``None``.

    Returns
    -------
    The inner content string, or ``None`` if input is ``None`` or malformed.
    """
    if boxed_str is None:
        return None
    # Identify the command name (everything before the first '{', stripped).
    brace_pos = boxed_str.find("{")
    if brace_pos == -1 or not boxed_str.endswith("}"):
        return None
    command = boxed_str[:brace_pos].rstrip()
    if command not in (r"\boxed", r"\fbox"):
        return None
    # Content sits between the opening '{' and the final '}'.
    return boxed_str[brace_pos + 1:-1]


def _unwrap_ground_truth(gt: str) -> str:
    """Unwrap ground truth if it is itself inside ``\\boxed`` or ``\\fbox``.

    Parameters
    ----------
    gt:
        Raw ground-truth string from the dataset.

    Returns
    -------
    Unwrapped string ready for ``grade_math500_answer``.
    """
    gt = str(gt).strip()
    if r"\boxed" in gt or r"\fbox" in gt:
        inner = _unwrap_box(last_boxed_only_string(gt))
        if inner is not None:
            return inner
    return gt


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

def extract_final_answer_strict(text: str) -> Optional[str]:
    """Extract the final answer using strict LaTeX-box matching only.

    This is the **official extractor** for the MATH-500 headline score.

    Priority order:

    1. Last ``\\boxed{...}`` in *text*.
    2. Last ``\\fbox{...}`` in *text* (fallback).
    3. ``None`` — no score awarded.

    Parameters
    ----------
    text:
        Raw model-generated response string.

    Returns
    -------
    Extracted answer string, or ``None`` if no boxed expression is found.
    """
    if not text:
        return None
    # last_boxed_only_string already searches \boxed first, then \fbox.
    return _unwrap_box(last_boxed_only_string(text))


# Plain-text fallback patterns tried in order when strict extraction fails.
# All line-anchored patterns use (?:^|\n) so they only match at the start of a
# line, preventing spurious matches on mid-sentence phrases like
# "the answer is: we need to compute ...".
_RELAXED_PATTERNS: list[tuple[str, re.Pattern]] = [
    # GSM8K delimiter  #### 42  (may appear mid-line, safe because #### is distinctive)
    ("hash4",        re.compile(r"####\s*(.+?)(?:\n|$)")),
    # "Final answer: ..."  or  "Final Answer - ..."  — must start a line
    ("final_answer", re.compile(r"(?:^|\n)[^\S\n]*[Ff]inal\s+[Aa]nswer\s*[:\-]\s*(.+?)(?:\n|$)")),
    # "Answer: ..."  — must start a line to avoid mid-sentence false positives
    ("answer_colon", re.compile(r"(?:^|\n)[^\S\n]*[Aa]nswer\s*[:\-]\s*(.+?)(?:\n|$)")),
]


def extract_final_answer_relaxed(text: str) -> Optional[str]:
    """Extract the final answer with relaxed, multi-pattern matching.

    **Diagnostic only.** Never report this as the headline MATH-500 score;
    use :func:`extract_final_answer_strict` for that.

    Priority order:

    1. :func:`extract_final_answer_strict` (preferred).
    2. ``#### <answer>``
    3. ``Final answer: <answer>``
    4. ``Answer: <answer>``
    5. ``None`` if all patterns fail.

    Parameters
    ----------
    text:
        Raw model-generated response string.

    Returns
    -------
    Extracted answer string, or ``None`` if all patterns fail.
    """
    # Always prefer strict extraction when available.
    strict = extract_final_answer_strict(text)
    if strict is not None:
        return strict

    if not text:
        return None

    for _, pattern in _RELAXED_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade_math500_answer(pred: str, gt: str) -> bool:
    """Grade a single extracted prediction against a ground-truth string.

    Applies the same two-stage comparison used by the training grader
    (mathd normalisation first, sympy equivalence second), but as a
    standalone eval helper with no dependency on ``compute_training_score``.

    Parameters
    ----------
    pred:
        Extracted answer string (already unwrapped from ``\\boxed``).
    gt:
        Ground-truth string (already unwrapped from ``\\boxed``).

    Returns
    -------
    ``True`` if the answer is judged correct, ``False`` otherwise.
    """
    return grade_answer_mathd(pred, gt) or grade_answer_sympy(pred, gt)


def compute_math500_eval_score(
    text: str,
    ground_truth: str,
    mode: str = "strict",
) -> float:
    """Score a single model response against a ground truth.

    This is the **top-level scoring function** for final MATH-500 evaluation.
    It is fully independent of ``compute_training_score``.

    Parameters
    ----------
    text:
        Raw model-generated response string.
    ground_truth:
        Ground-truth answer (may be wrapped in ``\\boxed{}``; will be
        unwrapped automatically).
    mode:
        ``"strict"``  — official headline score;
                        uses :func:`extract_final_answer_strict`.
        ``"relaxed"`` — diagnostic score;
                        uses :func:`extract_final_answer_relaxed`.

    Returns
    -------
    ``1.0`` if correct, ``0.0`` otherwise.

    Raises
    ------
    ValueError
        If *mode* is not ``"strict"`` or ``"relaxed"``.
    """
    if mode == "strict":
        pred = extract_final_answer_strict(text)
    elif mode == "relaxed":
        pred = extract_final_answer_relaxed(text)
    else:
        raise ValueError(f"mode must be 'strict' or 'relaxed', got {mode!r}")

    if pred is None:
        return 0.0

    gt = _unwrap_ground_truth(ground_truth)
    return 1.0 if grade_math500_answer(pred, gt) else 0.0
