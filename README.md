# OffTarget

**Drug repurposing candidate discovery via side-effect similarity.**

OffTarget is a Streamlit app that compares drugs by their side-effect
profiles, in the spirit of the [SIDER database](http://sideeffects.embl.de/)
(Side Effect Resource). The core idea: two drugs that cause a similar
*pattern* of side effects often act on the same underlying biology, even
when they're prescribed for completely unrelated diseases. That overlap can
surface repurposing candidates that indication-based search would never
suggest — this is, in fact, how several real drugs were discovered.

[Live app](https://github.com/vstimpson/OffTarget) · Built as a data science
portfolio project.

## How it works

1. **Fingerprint each drug.** Every drug is represented as a binary vector
   over a shared vocabulary of side-effect terms — `1` if the drug is
   documented to cause that effect, `0` otherwise.
2. **Compare fingerprints.** Similarity between two drugs is computed with:
   - **Jaccard index** — `|A ∩ B| / |A ∪ B|`, the fraction of the *combined*
     side-effect set the two drugs share. Penalizes drugs with very
     different total side-effect counts.
   - **Cosine similarity** — the cosine of the angle between the two binary
     vectors. Slightly more forgiving toward drugs with many reported
     effects.
3. **Rank.** Given a query drug, every other drug is ranked by similarity
   score, producing a top-N list of repurposing leads.
4. **Sanity check.** A handful of *already-documented* repurposing stories
   (Sildenafil, Minoxidil, Thalidomide) are used as ground truth: if the
   method is doing something reasonable, it should rank those drugs'
   real-world mechanistic relatives near the top of their own similarity
   lists. See the **Validated Case Studies** tab in the app.
5. **Visualize the structures.** Each drug's 3D structure is rendered
   alongside its similarity results, making the central point visible:
   side-effect similarity is a *phenotypic* signal, independent of chemical
   scaffold. The PDE5 inhibitors happen to look alike (they're a real
   chemical class); minoxidil and hydralazine don't, despite sharing a
   cardiovascular side-effect signature from the same mechanism.

## Project structure

```
OffTarget/
├── app.py                # Streamlit entrypoint (search UI, viz, case studies)
├── data_prep.py           # Parses raw data into a clean drug x side-effect matrix
├── similarity.py           # Jaccard/cosine similarity, top-N lookup
├── structures_prep.py     # Builds validated 3D conformers from SMILES
├── data/
│   ├── raw/
│   │   ├── demo_side_effects.csv   # curated demo dataset (default source)
│   │   └── drug_smiles.csv          # curated SMILES for 3D structures
│   ├── processed/
│   │   ├── drug_side_effect_matrix.parquet
│   │   └── drug_side_effect_matrix.csv
│   └── structures/        # generated .mol files (3D conformers) + properties.csv
├── assets/
│   ├── 3Dmol-min.js       # vendored 3Dmol.js viewer library (BSD-3-Clause)
│   └── 3Dmol-LICENSE.txt
├── .streamlit/
│   └── config.toml       # clinical teal/slate theme
├── requirements.txt
└── README.md
```

## Data source

The full [SIDER database](http://sideeffects.embl.de/downloads/) maps
~1,400 drugs to tens of thousands of side-effect terms extracted from
public drug labels. This project's development environment couldn't reach
`sideeffects.embl.de` directly (no outbound network access), so OffTarget
ships with a **curated demo dataset** instead:

- 61 drugs across diverse therapeutic classes (PDE5 inhibitors, SSRIs/SNRIs,
  statins, ACE inhibitors/ARBs, NSAIDs, opioids, anticonvulsants,
  antipsychotics, IMiDs, and more)
- 96 side-effect terms, hand-compiled from well-documented drug label data
- Every drug in the "Validated Case Studies" tab and its known mechanistic
  relatives, so the sanity check has something to recover

**To use the real SIDER database instead:** download `meddra_all_se.tsv.gz`
and `drug_names.tsv` from
[sideeffects.embl.de/downloads](http://sideeffects.embl.de/downloads/),
place them in `data/raw/`, and re-run:

```bash
python data_prep.py
```

`data_prep.py` automatically prefers the real SIDER files over the demo
dataset when both are present — no code changes needed.

## 3D structures

Each drug's 3D structure comes from a fully offline, one-time pipeline —
consistent with how the side-effect data is handled, and for the same
reason: this development environment couldn't reach PubChem or RCSB either.

1. `data/raw/drug_smiles.csv` holds a curated SMILES string and expected
   molecular formula for each drug, compiled from public chemical
   references.
2. `structures_prep.py` parses each SMILES with [RDKit](https://www.rdkit.org/),
   cross-checks the resulting molecular formula against the expected one
   (catching typos/invalid structures automatically), generates a 3D
   conformer (ETKDG embedding + MMFF94 optimization), and saves it to
   `data/structures/<drug>.mol`. It also computes a set of standard
   cheminformatics descriptors for each drug -- molecular weight, LogP,
   H-bond donor/acceptor counts, topological polar surface area (TPSA),
   rotatable bond count, and ring count -- and writes them to
   `data/structures/properties.csv`.
3. `app.py` renders the `.mol` files with [3Dmol.js](https://3dmol.org/),
   vendored locally in `assets/` (BSD-3-Clause) rather than loaded from a
   CDN, and displays each drug's formula and computed properties alongside
   its structure -- the viewer needs no network access at all, at build
   time or runtime.

**Accuracy caveat:** SMILES were compiled from training knowledge, not
cross-checked against a live structure database (same constraint as the
side-effect data). The automated formula check catches gross errors —
several genuinely wrong structures were caught and fixed this way during
development — but it doesn't guarantee full stereochemical or connectivity
correctness for more complex molecules. One drug in the dataset
(oxycodone) is excluded from this feature entirely because its structure
couldn't be confidently validated. Treat the 3D views as illustrative, not
as a certified structure database.

## Validated case studies

Three documented repurposing stories are used to sanity-check the method
(see the app's **Validated Case Studies** tab for the full write-ups and
live results):

| Drug | Originally developed for | Repurposed for | Known relatives recovered |
|---|---|---|---|
| **Sildenafil** (Viagra/Revatio) | Angina pectoris | Erectile dysfunction; pulmonary arterial hypertension | Tadalafil, Vardenafil (other PDE5 inhibitors) |
| **Minoxidil** (Loniten/Rogaine) | Severe hypertension | Androgenetic alopecia | Hydralazine (another direct vasodilator) |
| **Thalidomide** (Thalomid) | Sedative/antiemetic | Multiple myeloma; leprosy complications | Lenalidomide, Pomalidomide (IMiD analogs) |

In each case, the drug's real-world mechanistic relatives rank at or near
the top of its Jaccard similarity list, purely from shared side-effect
patterns — the app computes and displays this live rather than hardcoding
the result.

## Limitations

- The demo dataset is illustrative, not exhaustive. Absence of a shared
  side effect means it wasn't included in this curated list, not that it
  doesn't exist in reality.
- Side-effect co-occurrence is a *weak proxy* for shared mechanism — it can
  reflect genuine target overlap, but also coincidence, drug-class labeling
  conventions, or reporting bias in the source data.
- Results are hypothesis-generating leads for further investigation, not
  clinical or pharmacological conclusions.
- 3D structures are illustrative (see the caveat above) and one drug
  (oxycodone) has no 3D structure available.

## Running locally

Requires Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python data_prep.py               # build the processed matrix (auto-run by app.py if skipped)
streamlit run app.py
```

The generated 3D structures in `data/structures/` are already checked into
the repo, so no extra step is needed to see them. To regenerate them (e.g.
after editing `data/raw/drug_smiles.csv`), install RDKit separately — it's
a data-prep tool, not a runtime dependency, so it's not in
`requirements.txt` — and re-run the pipeline:

```bash
pip install rdkit
python structures_prep.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) by
pointing it at this repository's `app.py`. `requirements.txt` pins exact
package versions so the cloud environment matches local development.

## Tech stack

Python · Streamlit · pandas · NumPy · Plotly · PyArrow · RDKit (data prep) · 3Dmol.js
