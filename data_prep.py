"""Build the drug x side-effect fingerprint matrix used by SideMatch.

Two data sources are supported, tried in this order:

1. Real SIDER files, if present in data/raw/:
     - meddra_all_se.tsv.gz  (drug -> MedDRA side effect mappings)
     - drug_names.tsv        (STITCH compound id -> drug name)
   Download these from http://sideeffects.embl.de/downloads/ and drop them
   in data/raw/ to use the full SIDER database instead of the demo data.

2. The curated demo dataset, data/raw/demo_side_effects.csv, a hand-built
   long-format (drug_name, side_effect) table covering ~60 drugs across
   diverse therapeutic classes. This is the default so the app works with
   zero setup and no network access.

Either source is reshaped into the same output: a binary drug x side-effect
matrix, saved to data/processed/ as both parquet (fast load) and CSV
(portable/inspectable).

Run directly to (re)build the processed matrix:
    python data_prep.py
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SIDER_SE_FILE = RAW_DIR / "meddra_all_se.tsv.gz"
SIDER_NAMES_FILE = RAW_DIR / "drug_names.tsv"
DEMO_FILE = RAW_DIR / "demo_side_effects.csv"

MATRIX_PARQUET = PROCESSED_DIR / "drug_side_effect_matrix.parquet"
MATRIX_CSV = PROCESSED_DIR / "drug_side_effect_matrix.csv"


def load_sider_raw(se_path: Path = SIDER_SE_FILE,
                    names_path: Path = SIDER_NAMES_FILE) -> pd.DataFrame:
    """Parse the real SIDER TSVs into a long-format (drug_name, side_effect) table."""
    se_cols = [
        "stitch_id_flat", "stitch_id_stereo", "umls_concept_id_label",
        "meddra_concept_type", "umls_concept_id_meddra", "side_effect",
    ]
    se = pd.read_csv(se_path, sep="\t", names=se_cols, header=None)
    se = se[se["meddra_concept_type"] == "PT"]  # preferred terms only

    names = pd.read_csv(
        names_path, sep="\t", names=["stitch_id_flat", "drug_name"], header=None
    )

    merged = se.merge(names, on="stitch_id_flat", how="inner")
    merged["drug_name"] = merged["drug_name"].str.strip().str.title()
    merged["side_effect"] = merged["side_effect"].str.strip().str.lower()
    return merged[["drug_name", "side_effect"]].drop_duplicates()


def load_demo_dataset(path: Path = DEMO_FILE) -> pd.DataFrame:
    """Load the curated demo (drug_name, side_effect) table."""
    df = pd.read_csv(path)
    df["drug_name"] = df["drug_name"].str.strip()
    df["side_effect"] = df["side_effect"].str.strip().str.lower()
    return df.drop_duplicates()


def load_long_format() -> tuple[pd.DataFrame, str]:
    """Return (long_df, source_label), preferring real SIDER data if available."""
    if SIDER_SE_FILE.exists() and SIDER_NAMES_FILE.exists():
        return load_sider_raw(), "SIDER (full database)"
    return load_demo_dataset(), "curated demo dataset"


def build_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long (drug_name, side_effect) table into a binary drug x side-effect matrix."""
    long_df = long_df.copy()
    long_df["present"] = 1
    matrix = long_df.pivot_table(
        index="drug_name",
        columns="side_effect",
        values="present",
        fill_value=0,
        aggfunc="max",
    ).astype("int8")
    matrix = matrix.sort_index()
    return matrix


def save_matrix(matrix: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(MATRIX_PARQUET)
    matrix.to_csv(MATRIX_CSV)


def main() -> pd.DataFrame:
    long_df, source = load_long_format()
    matrix = build_matrix(long_df)
    save_matrix(matrix)
    print(
        f"Built matrix from {source}: {matrix.shape[0]} drugs x "
        f"{matrix.shape[1]} side effects -> {MATRIX_PARQUET}"
    )
    return matrix


if __name__ == "__main__":
    main()
