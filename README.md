# OffTarget

**Drug repurposing candidate discovery via side-effect similarity.**

OffTarget is a Streamlit app that compares drugs by their side-effect
profiles, in the spirit of the [SIDER database](http://sideeffects.embl.de/)
(Side Effect Resource). The core idea: two drugs that cause a similar
*pattern* of side effects often act on the same underlying biology, even
when they're prescribed for completely unrelated diseases. That overlap can
surface repurposing candidates that indication-based search would never
suggest. This is how several real drugs were discovered.

[Live app](https://github.com/vstimpson/OffTarget)

## Why this matters

Drug development is slow and expensive, but *repurposing* an
already-approved drug skips most of that cost. Its safety profile,
dosing, and manufacturing are already established, so a new indication can
go almost straight to efficacy trials instead of starting over. The catch
is finding good repurposing candidates in the first place: traditional
searches start from a known mechanism or indication, which means they can
only find what's already understood.

Side-effect similarity offers a cheap, data-driven alternative that needs
no prior knowledge of mechanism. It found Tadalafil and Vardenafil as
Sildenafil's relatives purely from overlapping side effects, without ever
being told "these are all PDE5 inhibitors", the kind of non-obvious lead
that indication or target-based search would miss entirely. The tradeoff
is that it's a hypothesis generator, not a validator: a high similarity
score means "worth investigating," not "will work." That's the reason the
**Validated Case Studies** tab exists, to confirm the method recovers
*known* good leads before trusting it on unknown ones.

## How it works

1. **Fingerprint each drug.** Every drug is represented as a binary vector
   over a shared vocabulary of side-effect terms. `1` if the drug is
   documented to cause that effect, `0` otherwise.
2. **Compare fingerprints.** Similarity between two drugs is computed with:
   - **Jaccard index** — `|A ∩ B| / |A ∪ B|`, the fraction of the *combined*
     side-effect set the two drugs share. Penalizes drugs with very
     different total side-effect counts.
   - **Cosine similarity** — the cosine of the angle between the two binary
     vectors. Slightly more forgiving toward drugs with many reported
     effects.

   Either metric can optionally be **IDF-weighted**: `w = log(N / n)` for
   each side effect (N = total drugs, n = drugs with that side effect), the
   same idea as IDF in TF-IDF applied to a presence matrix instead of word
   counts. Headache and nausea, present in most drugs, end up near weight
   0; a side effect present in one drug out of sixty gets the highest
   weight. See **Does IDF weighting help?** below — it isn't just a
   plausible tweak, it measurably improves agreement with known drug
   targets.
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
6. **Flag off-target hypotheses.** Every match is checked against a curated
   table of known drug targets. A match that already shares a known target
   confirms the method is working; a high-similarity match with *no* known
   shared target is flagged as an off-target hypothesis — the more
   interesting case, since it points at a possible mechanism nobody's
   documented yet. See **Off-target hypotheses** below.

## Project structure

```
OffTarget/
├── app.py                # Streamlit entrypoint (search UI, viz, case studies)
├── data_prep.py           # Parses raw data into a clean drug x side-effect matrix
├── similarity.py           # Jaccard/cosine similarity, top-N lookup
├── structures_prep.py     # Builds validated 3D conformers from SMILES
├── targets.py             # Known-target lookup + off-target/reframing hypothesis scans
├── data/
│   ├── raw/
│   │   ├── demo_side_effects.csv         # curated demo dataset (default source)
│   │   ├── drug_smiles.csv                # curated SMILES for 3D structures
│   │   ├── drug_targets.csv               # curated known primary target per drug
│   │   └── side_effect_reframings.csv     # side effects with precedent as a drug's actual purpose
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

## Off-target hypotheses

This is the part of the methodology that gives the app its name. It follows
[Campillos et al., "Drug target identification using side-effect
similarity"](https://doi.org/10.1126/science.1158140) (*Science*, 2008) —
the paper this whole approach is built on. Its key move: side-effect
similarity isn't just useful for finding a drug's new *indication*, it can
predict a drug's molecular **target**, including targets nobody has linked
that drug to yet. The authors validated several such predictions
experimentally (in vitro binding assays).

OffTarget applies that logic directly:

- `data/raw/drug_targets.csv` curates each drug's known primary molecular
  target and a broader "target family" string used for comparison (e.g.
  every statin gets `HMG-CoA reductase`; every PDE5 inhibitor gets
  `Phosphodiesterase type 5 (PDE5)`).
- `targets.py` classifies every drug pair as **shared** (same target
  family — the method found something already known), **off-target** (high
  side-effect similarity, no known shared target — a genuine hypothesis),
  or **unknown** (target not curated).
- The **Search** tab badges each result accordingly; the **Off-Target
  Hypotheses** tab scans the entire dataset for the strongest off-target
  pairs at an adjustable similarity threshold, with each pair's 3D
  structures and computed properties one click away.

**Caveat:** target curation is a simplification. Some drugs are
polypharmacological (e.g. tramadol, valproate) and are bucketed under a
single dominant/representative target family; two drugs with genuinely
related but non-identical targets (e.g. a beta-1-selective vs.
non-selective beta blocker) may be flagged "off-target" even though they're
mechanistically close. These are computational leads, not findings — real
off-target hypotheses need experimental validation before they mean
anything clinically, same as in the original paper.

### Reframed side effects

A second, more literal reading of "bad side effects, used positively": some
side effects have real precedent for becoming a drug's actual therapeutic
purpose — sildenafil's priapism became Viagra; minoxidil's hypertrichosis
became Rogaine. `data/raw/side_effect_reframings.csv` curates a handful of
these precedents (including a third: topiramate's and bupropion's
weight-loss side effect, deliberately turned into the weight-management
drugs Qsymia and Contrave). `targets.py` then generalizes each one past its
single pioneer drug — for any drug that shares that same side effect, it's
flagged as an untapped candidate for the same reframed purpose. The
**Search** tab surfaces this for whichever drug you're looking at; the
**Off-Target Hypotheses** tab lists every candidate across the whole
dataset.

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

### Does this hold up across the whole dataset?

Three case studies are hand-picked by definition. The same tab also runs a
dataset-wide version of the same test: for every drug pair with a curated
target (`targets.all_pairs_target_overlap`), does higher side-effect
similarity correspond to a higher rate of actually sharing a known target?

Across all 1,830 pairs in the current demo dataset: pairs above 0.5 Jaccard
similarity share a known target 51.9% of the time, versus under 1% for
pairs below 0.1 — a Pearson correlation of 0.41 between similarity and
shared-target status. That's the general-case evidence behind the three
anecdotes: the method is picking up something real across the dataset, not
just for three examples chosen because they already work. It's also not
evidence the method is reliable for any *single* pair — most high-similarity
pairs still don't share a known target, which is precisely the off-target
hypothesis space the next section is about.

### Does IDF weighting help?

The 0.41 correlation above uses plain Jaccard, where every side effect
counts equally. `similarity.py`'s IDF weighting (`w = log(N / n)`, rare
side effects weighted higher than common ones) is a specific, testable
claim: it should make similarity track known targets *more* closely. Same
1,830 pairs, same target curation, only the weighting changes:

| | Correlation (similarity vs. shared target) |
|---|---|
| Unweighted Jaccard | 0.41 |
| IDF-weighted Jaccard | 0.51 |

The Validated Case Studies tab shows both numbers live, side by side, so
this isn't a claim you have to take on faith — the toggle recomputes it in
the browser. IDF weighting is on by default in the Search and Off-Target
Hypotheses tabs, with an option to switch it off and see the naive
baseline for comparison.

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
- Known-target curation is a simplification (see the caveat above); an
  "off-target hypothesis" badge means *no known shared target in this
  dataset*, not an experimentally confirmed novel mechanism.
- IDF weights are computed from a 61-drug demo dataset, not the full ~1,400
  SIDER catalog, so `n_i` for any given side effect is a small, noisy count
  — a side effect that looks "rare" here might not be rare in reality. The
  weighted-vs-unweighted correlation comparison would be worth re-running
  once real SIDER data is plugged in.

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
