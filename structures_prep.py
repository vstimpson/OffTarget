"""Build 3D conformers for the drugs in the OffTarget demo dataset.

Reads data/raw/drug_smiles.csv (drug_name, smiles, expected_formula),
validates each structure against its expected molecular formula, embeds a
3D conformer, and writes it to data/structures/<drug>.mol.

This is a one-time, offline data-prep step -- consistent with how
data_prep.py treats the side-effect data: fetched/curated once, stored as a
static file, no network calls from the running app.
"""

import csv
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

RAW_PATH = Path("data/raw/drug_smiles.csv")
OUTPUT_DIR = Path("data/structures")


def _formula_counter(formula: str) -> Counter:
    formula = re.sub(r"[+-]\d*$", "", formula)
    return Counter(
        {el: int(n) if n else 1 for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if el}
    )


def build_structures() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    built, skipped = 0, []

    with RAW_PATH.open() as fh:
        for row in csv.DictReader(fh):
            name, smiles, expected = row["drug_name"], row["smiles"], row["expected_formula"]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped.append((name, "invalid SMILES"))
                continue

            actual = rdMolDescriptors.CalcMolFormula(mol)
            if _formula_counter(actual) != _formula_counter(expected):
                skipped.append((name, f"formula mismatch: expected {expected}, got {actual}"))
                continue

            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                skipped.append((name, "3D embedding failed"))
                continue
            AllChem.MMFFOptimizeMolecule(mol)

            out_path = OUTPUT_DIR / f"{name}.mol"
            Chem.MolToMolFile(mol, str(out_path))
            built += 1

    print(f"Built {built} 3D structures in {OUTPUT_DIR}/")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    build_structures()
