"""Known drug-target data and off-target hypothesis logic for OffTarget.

Grounded in Campillos et al., "Drug target identification using
side-effect similarity" (Science, 2008): side-effect similarity was
originally proposed not just to find repurposing candidates, but to
predict shared molecular targets -- including previously-unknown
("off-target") ones. A high-similarity drug pair that already shares a
known target confirms the method is picking up real biology; a
high-similarity pair with no known shared target is a genuine off-target
hypothesis worth investigating.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pathways import pathway_relationship

TARGETS_PATH = Path("data/raw/drug_targets.csv")
REFRAMINGS_PATH = Path("data/raw/side_effect_reframings.csv")


def load_targets(path: Path = TARGETS_PATH) -> pd.DataFrame:
    return pd.read_csv(path).set_index("drug_name")


def load_reframings(path: Path = REFRAMINGS_PATH) -> pd.DataFrame:
    """Side effects with real historical precedent for becoming the actual
    therapeutic purpose of a drug (e.g. sildenafil's priapism -> Viagra).
    """
    return pd.read_csv(path)


def drug_reframing_signals(drug: str, matrix: pd.DataFrame, reframings: pd.DataFrame) -> pd.DataFrame:
    """Which of `drug`'s side effects have precedent for becoming a purpose.

    Returns rows with side_effect, reframed_purpose, pioneer_drug, note, and
    is_pioneer (True if `drug` itself is the drug that precedent came from).
    """
    if drug not in matrix.index:
        return reframings.iloc[0:0]
    drug_effects = set(matrix.columns[matrix.loc[drug] == 1])
    hits = reframings[reframings["side_effect"].isin(drug_effects)].copy()
    hits["is_pioneer"] = hits["pioneer_drug"] == drug
    return hits.reset_index(drop=True)


def reframing_candidates(matrix: pd.DataFrame, reframings: pd.DataFrame) -> pd.DataFrame:
    """Scan the whole dataset: for every curated reframing, which drugs (other
    than the pioneer) also share that side effect and are untapped candidates
    for the same reframed purpose?
    """
    rows = []
    for _, ref in reframings.iterrows():
        side_effect = ref["side_effect"]
        if side_effect not in matrix.columns:
            continue
        carriers = matrix.index[matrix[side_effect] == 1]
        for drug in carriers:
            if drug == ref["pioneer_drug"]:
                continue
            rows.append(
                {
                    "candidate_drug": drug,
                    "side_effect": side_effect,
                    "reframed_purpose": ref["reframed_purpose"],
                    "pioneer_drug": ref["pioneer_drug"],
                }
            )
    columns = ["candidate_drug", "side_effect", "reframed_purpose", "pioneer_drug"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)


def target_relationship(drug_a: str, drug_b: str, targets: pd.DataFrame) -> str:
    """Classify the target relationship between two drugs.

    "shared" -- both have a curated target in the same family.
    "off_target" -- both are curated but the families differ.
    "unknown" -- either drug has no curated target.
    """
    if drug_a not in targets.index or drug_b not in targets.index:
        return "unknown"
    fam_a = targets.loc[drug_a, "target_family"]
    fam_b = targets.loc[drug_b, "target_family"]
    return "shared" if fam_a == fam_b else "off_target"


def top_off_target_hypotheses(
    sims: pd.DataFrame,
    targets: pd.DataFrame,
    min_similarity: float = 0.25,
    n: int = 15,
) -> pd.DataFrame:
    """Scan the full similarity matrix for the strongest off-target hypotheses:
    drug pairs with high side-effect similarity but no known shared target.
    """
    seen: set[frozenset[str]] = set()
    rows = []
    for drug_a in sims.index:
        for drug_b, score in sims.loc[drug_a].items():
            if drug_a == drug_b or score < min_similarity:
                continue
            pair = frozenset((drug_a, drug_b))
            if pair in seen:
                continue
            seen.add(pair)
            if target_relationship(drug_a, drug_b, targets) != "off_target":
                continue
            rows.append(
                {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "similarity": round(float(score), 4),
                    "target_a": targets.loc[drug_a, "primary_target"],
                    "target_b": targets.loc[drug_b, "primary_target"],
                    "pathway_relationship": pathway_relationship(drug_a, drug_b, targets),
                }
            )

    columns = ["drug_a", "drug_b", "similarity", "target_a", "target_b", "pathway_relationship"]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values("similarity", ascending=False).head(n).reset_index(drop=True)


def all_pairs_target_overlap(sims: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Every drug pair with a curated target on both sides: similarity score
    and whether they share a target family. The general-purpose validation
    dataset -- does side-effect similarity correspond to target overlap,
    across the whole matrix, not just hand-picked examples?
    """
    seen: set[frozenset[str]] = set()
    rows = []
    for drug_a in sims.index:
        if drug_a not in targets.index:
            continue
        for drug_b, score in sims.loc[drug_a].items():
            if drug_a == drug_b or drug_b not in targets.index:
                continue
            pair = frozenset((drug_a, drug_b))
            if pair in seen:
                continue
            seen.add(pair)
            relationship = pathway_relationship(drug_a, drug_b, targets)
            rows.append(
                {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "similarity": float(score),
                    "shares_target": relationship == "shared_target",
                    "shares_pathway": relationship in ("shared_target", "shared_pathway"),
                    "target_a": targets.loc[drug_a, "primary_target"],
                    "target_b": targets.loc[drug_b, "primary_target"],
                }
            )
    columns = [
        "drug_a", "drug_b", "similarity", "shares_target", "shares_pathway",
        "target_a", "target_b",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def target_overlap_correlation(pairs: pd.DataFrame, column: str = "shares_target") -> dict:
    """Summary stats for the similarity-vs-shared-target (or -pathway)
    relationship. Pass column="shares_pathway" for the pathway-level version.
    """
    if pairs.empty:
        return {
            "n_pairs": 0,
            "n_shared": 0,
            "correlation": float("nan"),
            "mean_sim_shared": float("nan"),
            "mean_sim_not_shared": float("nan"),
        }
    shared = pairs[column]
    return {
        "n_pairs": len(pairs),
        "n_shared": int(shared.sum()),
        "correlation": float(pairs["similarity"].corr(shared.astype(int))),
        "mean_sim_shared": float(pairs.loc[shared, "similarity"].mean()) if shared.any() else float("nan"),
        "mean_sim_not_shared": float(pairs.loc[~shared, "similarity"].mean()) if (~shared).any() else float("nan"),
    }


def target_overlap_by_similarity_bin(
    pairs: pd.DataFrame, bin_edges: list[float], column: str = "shares_target"
) -> pd.DataFrame:
    """Bin pairs by similarity and compute the share-a-target (or -pathway)
    rate per bin -- the chart that answers "does higher similarity mean more
    shared biology?" Pass column="shares_pathway" for the pathway-level version.
    """
    if pairs.empty:
        return pd.DataFrame(columns=["bin_label", "pct_shared", "n_pairs"])
    bins = pd.cut(pairs["similarity"], bins=bin_edges, include_lowest=True)
    summary = pairs.groupby(bins, observed=True)[column].agg(["mean", "count"])
    summary = summary.reset_index()
    summary.columns = ["bin", "pct_shared", "n_pairs"]
    summary["bin_label"] = summary["bin"].apply(lambda b: f"{max(b.left, 0):.1f}-{b.right:.1f}")
    return summary[["bin_label", "pct_shared", "n_pairs"]]
