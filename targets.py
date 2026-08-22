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

TARGETS_PATH = Path("data/raw/drug_targets.csv")


def load_targets(path: Path = TARGETS_PATH) -> pd.DataFrame:
    return pd.read_csv(path).set_index("drug_name")


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
                }
            )

    columns = ["drug_a", "drug_b", "similarity", "target_a", "target_b"]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values("similarity", ascending=False).head(n).reset_index(drop=True)
