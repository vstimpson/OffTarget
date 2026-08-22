"""Cross-cutting analysis for OffTarget: surprising pairs, a repurposing
score, 2D projections, and clustering of the side-effect fingerprint space.

Everything here draws on similarity.py (the fingerprints and similarity
scores) plus targets.py-style curated data (therapeutic category), so it
lives in its own module rather than bloating either.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from targets import target_relationship

CATEGORIES_PATH = Path("data/raw/drug_categories.csv")

BIOLOGICAL_PLAUSIBILITY = {"shared": 1.0, "off_target": 0.5, "unknown": 0.3}
INDICATION_DIFFERENCE = {True: 1.0, False: 0.2}


def load_categories(path: Path = CATEGORIES_PATH) -> pd.DataFrame:
    return pd.read_csv(path).set_index("drug_name")


def repurposing_score(similarity: float, relationship: str, category_differs: bool) -> float:
    """Exploratory composite score -- NOT a validated clinical metric.

    similarity: side-effect similarity (0-1).
    relationship: "shared" / "off_target" / "unknown" target status, used
      as a stand-in for "biological plausibility" -- a confirmed shared
      target is the most plausible, an off-target hypothesis is plausible
      but unconfirmed, unknown is the least assessable.
    category_differs: whether the two drugs' therapeutic categories differ
      -- a repurposing lead is more interesting the further apart the two
      drugs' current uses are.
    """
    plausibility = BIOLOGICAL_PLAUSIBILITY[relationship]
    indication_diff = INDICATION_DIFFERENCE[category_differs]
    return similarity * plausibility * indication_diff


def surprising_pairs(
    sims: pd.DataFrame,
    categories: pd.DataFrame,
    targets: pd.DataFrame,
    min_similarity: float = 0.3,
    n: int = 25,
    sort_by: str = "similarity",
) -> pd.DataFrame:
    """Rank drug pairs that are highly similar by side effects but belong to
    different therapeutic categories -- the "surprising pairs" investigation.

    sort_by: "similarity" (default -- the literal ask: most similar first)
    or "repurposing_score" (weights toward pairs that also have biological
    plausibility and a bigger indication gap; see repurposing_score()).
    """
    seen: set[frozenset[str]] = set()
    rows = []
    for drug_a in sims.index:
        if drug_a not in categories.index:
            continue
        for drug_b, score in sims.loc[drug_a].items():
            if drug_a == drug_b or drug_b not in categories.index or score < min_similarity:
                continue
            pair = frozenset((drug_a, drug_b))
            if pair in seen:
                continue
            seen.add(pair)

            cat_a = categories.loc[drug_a, "therapeutic_category"]
            cat_b = categories.loc[drug_b, "therapeutic_category"]
            if cat_a == cat_b:
                continue  # not "surprising" -- same category, similarity is expected

            relationship = target_relationship(drug_a, drug_b, targets)
            score_f = float(score)
            rows.append(
                {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "similarity": round(score_f, 4),
                    "category_a": cat_a,
                    "category_b": cat_b,
                    "target_relationship": relationship,
                    "target_a": targets.loc[drug_a, "primary_target"] if drug_a in targets.index else "unknown",
                    "target_b": targets.loc[drug_b, "primary_target"] if drug_b in targets.index else "unknown",
                    "repurposing_score": round(
                        repurposing_score(score_f, relationship, True), 4
                    ),
                }
            )

    columns = [
        "drug_a", "drug_b", "similarity", "category_a", "category_b",
        "target_relationship", "target_a", "target_b", "repurposing_score",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    sort_col = "repurposing_score" if sort_by == "repurposing_score" else "similarity"
    return result.sort_values(sort_col, ascending=False).head(n).reset_index(drop=True)


def pca_projection(matrix: pd.DataFrame, n_components: int = 2, random_state: int = 42) -> pd.DataFrame:
    """2D PCA projection of the drug x side-effect fingerprint matrix."""
    coords = PCA(n_components=n_components, random_state=random_state).fit_transform(matrix.values)
    return pd.DataFrame(coords, index=matrix.index, columns=["x", "y"])


def tsne_projection(
    matrix: pd.DataFrame, perplexity: float = 15.0, random_state: int = 42
) -> pd.DataFrame:
    """2D t-SNE projection. Perplexity is clamped to stay valid for small N."""
    n = len(matrix)
    perplexity = min(perplexity, max(5.0, (n - 1) / 3))
    coords = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state, init="pca"
    ).fit_transform(matrix.values.astype(np.float64))
    return pd.DataFrame(coords, index=matrix.index, columns=["x", "y"])


def cluster_drugs(matrix: pd.DataFrame, n_clusters: int, method: str = "kmeans") -> pd.Series:
    """Cluster drugs by side-effect fingerprint only (no target/category
    info) -- KMeans or agglomerative (hierarchical) clustering.
    """
    if method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(matrix.values)
    elif method == "hierarchical":
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(matrix.values)
    else:
        raise ValueError(f"method must be 'kmeans' or 'hierarchical', got {method!r}")
    return pd.Series(labels, index=matrix.index, name="cluster")


def cluster_category_purity(clusters: pd.Series, categories: pd.DataFrame) -> pd.DataFrame:
    """For each cluster, how concentrated is it in a single therapeutic
    category? A high-purity cluster recovered known drug classes purely
    from side effects, with no indication data involved in forming it.
    """
    df = pd.DataFrame({"cluster": clusters})
    df = df.join(categories, how="left")
    rows = []
    for cluster_id, group in df.groupby("cluster"):
        counts = group["therapeutic_category"].value_counts()
        top_category = counts.index[0] if len(counts) else "unknown"
        purity = counts.iloc[0] / len(group) if len(group) else 0.0
        rows.append(
            {
                "cluster": cluster_id,
                "size": len(group),
                "top_category": top_category,
                "purity": round(float(purity), 3),
                "drugs": ", ".join(sorted(group.index)),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
