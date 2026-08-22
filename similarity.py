"""Side-effect similarity logic for OffTarget.

Drugs are represented as binary side-effect fingerprints (one row per drug in
the matrix built by data_prep.py). Similarity between two fingerprints is
computed with Jaccard index (intersection over union of side effects) or
cosine similarity, both vectorized across the whole matrix at once.

Both metrics can optionally be IDF-weighted: side effects that appear in
almost every drug (headache, nausea, dizziness) carry little information
for distinguishing drugs and are down-weighted, while rare side effects
count for more -- the same idea as TF-IDF in text retrieval, applied to a
binary presence matrix instead of word counts.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import MATRIX_PARQUET, main as build_matrix_file

VALID_METRICS = ("jaccard", "cosine")


def load_matrix(path: Path = MATRIX_PARQUET) -> pd.DataFrame:
    """Load the drug x side-effect binary matrix, building it first if missing."""
    if not path.exists():
        build_matrix_file()
    return pd.read_parquet(path)


def jaccard_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard similarity (intersection / union) between every drug."""
    values = matrix.values.astype(np.float64)
    intersection = values @ values.T
    row_sums = values.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def cosine_similarity_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine similarity between every drug's fingerprint vector."""
    values = matrix.values.astype(np.float64)
    norms = np.linalg.norm(values, axis=1)
    norms_safe = np.where(norms == 0, 1.0, norms)
    normalized = values / norms_safe[:, None]
    sim = normalized @ normalized.T
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def idf_weights(matrix: pd.DataFrame) -> pd.Series:
    """IDF-style informativeness weight per side effect: w_i = log(N / n_i).

    N is the number of drugs; n_i is the number of drugs documented to
    cause side effect i. Side effects shared by nearly every drug (headache,
    nausea, dizziness) score close to 0 -- they don't help distinguish
    drugs. Rare side effects score higher and count for more.
    """
    n_drugs = len(matrix)
    n_i = matrix.sum(axis=0).clip(lower=1)
    return np.log(n_drugs / n_i)


def apply_weights(matrix: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Scale each side-effect column by its weight."""
    return matrix * weights.reindex(matrix.columns).to_numpy()


def weighted_jaccard_similarity_matrix(matrix: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """IDF-weighted (Ruzicka) Jaccard similarity.

    For binary vectors, sum(min(w*a, w*b)) / sum(max(w*a, w*b)) reduces to
    the sum of weights in the shared side effects over the sum of weights
    in the combined side effects -- the natural weighted generalization of
    "intersection over union."
    """
    values = matrix.values.astype(np.float64)
    w = weights.reindex(matrix.columns).to_numpy(dtype=np.float64)
    weighted_values = values * w
    intersection = weighted_values @ values.T
    row_w_sums = weighted_values.sum(axis=1)
    union = row_w_sums[:, None] + row_w_sums[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)


def similarity_matrix(
    matrix: pd.DataFrame,
    metric: str = "jaccard",
    weighted: bool = False,
) -> pd.DataFrame:
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")
    if not weighted:
        if metric == "jaccard":
            return jaccard_similarity_matrix(matrix)
        return cosine_similarity_matrix(matrix)

    weights = idf_weights(matrix)
    if metric == "jaccard":
        return weighted_jaccard_similarity_matrix(matrix, weights)
    return cosine_similarity_matrix(apply_weights(matrix, weights))


def explain_similarity(
    drug_a: str,
    drug_b: str,
    matrix: pd.DataFrame,
    weighted: bool = True,
    top_k: int = 5,
) -> pd.DataFrame:
    """Break a pair's similarity down into its shared side effects, ranked by
    how much each one actually contributes -- the "why" behind a score.

    Returns columns: side_effect, weight, pct_of_shared_weight, sorted by
    weight descending. With weighted=True, "weight" is the IDF weight
    (bigger = rarer = more informative); with weighted=False every shared
    side effect gets weight 1 and the ranking is arbitrary but the shared
    set itself is still useful to show.
    """
    if drug_a not in matrix.index or drug_b not in matrix.index:
        return pd.DataFrame(columns=["side_effect", "weight", "pct_of_shared_weight"])

    shared = matrix.columns[(matrix.loc[drug_a] == 1) & (matrix.loc[drug_b] == 1)]
    if len(shared) == 0:
        return pd.DataFrame(columns=["side_effect", "weight", "pct_of_shared_weight"])

    weights = idf_weights(matrix) if weighted else pd.Series(1.0, index=matrix.columns)
    contrib = weights.loc[shared].sort_values(ascending=False)
    total = contrib.sum()
    pct = (contrib / total * 100) if total > 0 else contrib * 0

    result = pd.DataFrame(
        {
            "side_effect": contrib.index,
            "weight": contrib.values.round(3),
            "pct_of_shared_weight": pct.values.round(1),
        }
    ).reset_index(drop=True)
    return result.head(top_k) if top_k else result


def resolve_drug_name(query: str, available: list[str]) -> str | None:
    """Case-insensitive exact match, falling back to the closest fuzzy match."""
    lookup = {name.lower(): name for name in available}
    if query.lower() in lookup:
        return lookup[query.lower()]
    close = difflib.get_close_matches(query, available, n=1, cutoff=0.6)
    return close[0] if close else None


def suggest_drug_names(query: str, available: list[str], n: int = 5) -> list[str]:
    """Fuzzy suggestions for a query that didn't resolve to an exact drug."""
    return difflib.get_close_matches(query, available, n=n, cutoff=0.3)


def top_n_similar(
    drug_name: str,
    matrix: pd.DataFrame,
    n: int = 10,
    metric: str = "jaccard",
    weighted: bool = False,
) -> pd.DataFrame:
    """Return the top-N drugs most similar to `drug_name` by side-effect profile.

    Output columns: drug_name, similarity, shared_side_effects, n_shared.
    Raises ValueError if the drug isn't found (with close-match suggestions).
    """
    resolved = resolve_drug_name(drug_name, list(matrix.index))
    if resolved is None:
        suggestions = suggest_drug_names(drug_name, list(matrix.index))
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown drug '{drug_name}'.{hint}")

    sims = similarity_matrix(matrix, metric=metric, weighted=weighted)
    scores = sims.loc[resolved].drop(resolved).sort_values(ascending=False)

    query_effects = set(matrix.columns[matrix.loc[resolved] == 1])
    rows = []
    for other, score in scores.head(n).items():
        other_effects = set(matrix.columns[matrix.loc[other] == 1])
        shared = sorted(query_effects & other_effects)
        rows.append(
            {
                "drug_name": other,
                "similarity": round(float(score), 4),
                "shared_side_effects": ", ".join(shared),
                "n_shared": len(shared),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    m = load_matrix()
    print(f"Loaded matrix: {m.shape[0]} drugs x {m.shape[1]} side effects\n")
    for query in ["Sildenafil", "Minoxidil", "Thalidomide"]:
        print(f"Top 5 similar to {query} (Jaccard):")
        print(top_n_similar(query, m, n=5, metric="jaccard").to_string(index=False))
        print()
