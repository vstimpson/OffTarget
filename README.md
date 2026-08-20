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

## Project structure

```
OffTarget/
├── app.py              # Streamlit entrypoint (search UI, viz, case studies)
├── data_prep.py         # Parses raw data into a clean drug x side-effect matrix
├── similarity.py         # Jaccard/cosine similarity, top-N lookup
├── data/
│   ├── raw/
│   │   └── demo_side_effects.csv   # curated demo dataset (default source)
│   └── processed/
│       ├── drug_side_effect_matrix.parquet
│       └── drug_side_effect_matrix.csv
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

## Running locally

Requires Python 3.11.

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python data_prep.py               # build the processed matrix (auto-run by app.py if skipped)
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) by
pointing it at this repository's `app.py`. `requirements.txt` pins exact
package versions so the cloud environment matches local development.

## Tech stack

Python · Streamlit · pandas · NumPy · Plotly · PyArrow
