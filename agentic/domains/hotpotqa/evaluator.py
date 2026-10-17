"""Answer metrics identical to dspy.evaluate.answer_exact_match / answer_passage_match helpers."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_exact_match(prediction: str, answers: list[str] | str) -> bool:
    answers = [answers] if isinstance(answers, str) else list(answers)
    return any(em_score(prediction, a) for a in answers)


def gold_title_recall(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    if not gold_titles:
        return 0.0
    got = {t.strip().lower() for t in retrieved_titles}
    return sum(1 for g in gold_titles if g.strip().lower() in got) / len(gold_titles)
