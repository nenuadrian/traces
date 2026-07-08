"""Metric helpers (dependency-free)."""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence


def ari(a: Sequence[int], b: Sequence[int]) -> float:
    """Adjusted Rand Index between two labelings."""
    n = len(a)
    if n == 0:
        return 1.0
    pair = Counter(zip(a, b))
    ca = Counter(a)
    cb = Counter(b)

    def comb2(x: int) -> float:
        return x * (x - 1) / 2.0

    sum_ij = sum(comb2(c) for c in pair.values())
    sum_a = sum(comb2(c) for c in ca.values())
    sum_b = sum(comb2(c) for c in cb.values())
    total = comb2(n)
    if total == 0:
        return 1.0
    expected = sum_a * sum_b / total
    max_index = (sum_a + sum_b) / 2.0
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


def mae(a: List[float], b: List[float]) -> float:
    assert len(a) == len(b)
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0
