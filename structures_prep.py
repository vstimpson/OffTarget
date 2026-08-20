"""Build 3D conformers and property tables for the OffTarget demo drugs.

Reads data/raw/drug_smiles.csv (drug_name, smiles, expected_formula),
validates each structure against its expected molecular formula, embeds a
3D conformer, and writes it to data/structures/<drug>.mol. Also computes a
handful of standard cheminformatics descriptors (molecular weight, LogP,
H-bond donors/acceptors, TPSA, rotatable bonds, ring count) and writes them
to data/structures/properties.csv for the app to display.

This is a one-time, offline data-prep step -- consistent with how
data_prep.py treats the side-effect data: fetched/curated once, stored as a
static file, no network calls from the running app.
"""

import csv
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RAW_PATH = Path("data/raw/drug_smiles.csv")
OUTPUT_DIR = Path("data/structures")
PROPERTIES_PATH = OUTPUT_DIR / "properties.csv"
PROPERTY_FIELDS = [
    "drug_name",
    "molecular_formula",
    "molecular_weight",
    "logp",
    "h_bond_donors",
    "h_bond_acceptors",
    "tpsa",
    "rotatable_bonds",
    "ring_count",
]


def _formula_counter(formula: str) -> Counter:
    formula = re.sub(r"[+-]\d*$", "", formula)
    return Counter(
        {el: int(n) if n else 1 for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if el}
    )


def _compute_properties(name: str, mol: Chem.Mol, formula: str) -> dict:
    return {
        "drug_name": name,
        "molecular_formula": formula,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "h_bond_donors": Descriptors.NumHDonors(mol),
        "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
    }


def build_structures() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    built, skipped = 0, []
    properties = []

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

            properties.append(_compute_properties(name, mol, actual))

            mol_3d = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_3d, randomSeed=42) != 0:
                skipped.append((name, "3D embedding failed"))
                continue
            AllChem.MMFFOptimizeMolecule(mol_3d)

            out_path = OUTPUT_DIR / f"{name}.mol"
            Chem.MolToMolFile(mol_3d, str(out_path))
            built += 1

    with PROPERTIES_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PROPERTY_FIELDS)
        writer.writeheader()
        writer.writerows(properties)

    print(f"Built {built} 3D structures in {OUTPUT_DIR}/")
    print(f"Wrote properties for {len(properties)} drugs to {PROPERTIES_PATH}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    build_structures()
