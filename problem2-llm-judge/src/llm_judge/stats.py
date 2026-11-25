"""Small, dependency-free statistics used for validation and A/B decisions."""

import math
import random
from collections import Counter


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stdev(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs, ys) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = sum(p[0] for p in pairs) / len(pairs)
    my = sum(p[1] for p in pairs) / len(pairs)
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs))
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pairs))
    if sx == 0 or sy == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pairs) / (sx * sy)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs, ys) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    return pearson(_ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs]))


def cohen_kappa(a: list, b: list, weights: str | None = None, categories: list | None = None) -> float | None:
    """Cohen's kappa between two raters. weights=None (nominal) or 'linear' (ordinal categories,
    which must then be given in order)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    cats = categories or sorted({x for p in pairs for x in p})
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(pairs)

    def w(i, j):
        if weights == "linear":
            return 1 - abs(i - j) / (k - 1) if k > 1 else 1.0
        return 1.0 if i == j else 0.0

    obs = Counter((idx[x], idx[y]) for x, y in pairs)
    pa = Counter(idx[x] for x, _ in pairs)
    pb = Counter(idx[y] for _, y in pairs)
    po = sum(w(i, j) * c for (i, j), c in obs.items()) / n
    pe = sum(w(i, j) * pa[i] * pb[j] for i in range(k) for j in range(k)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial (sign) test on non-tied pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: list[float], iters: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float] | None:
    diffs = [d for d in diffs if d is not None]
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters) - 1]


def entropy(counts: dict) -> float:
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c) if n else 0.0
