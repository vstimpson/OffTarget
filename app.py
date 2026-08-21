"""OffTarget: drug repurposing candidate finder via side-effect similarity.

Streamlit entrypoint. Compares drugs' side-effect "fingerprints" (SIDER-style
binary vectors) to surface repurposing leads -- drugs used for unrelated
conditions that nonetheless share a side-effect signature, hinting at a
shared underlying biological mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from similarity import load_matrix, similarity_matrix, top_n_similar
from targets import load_targets, target_relationship, top_off_target_hypotheses

CLINICAL_SCALE = ["#F8FAFC", "#CCFBF1", "#5EEAD4", "#0D9488", "#134E4A"]
STRUCTURES_DIR = Path("data/structures")
VIEWER_JS_PATH = Path("assets/3Dmol-min.js")
PROPERTIES_PATH = STRUCTURES_DIR / "properties.csv"

CASE_STUDIES = [
    {
        "drug": "Sildenafil",
        "brand": "Viagra / Revatio",
        "original_use": "Angina pectoris (coronary artery disease)",
        "repurposed_use": "Erectile dysfunction; pulmonary arterial hypertension",
        "known_relatives": ["Tadalafil", "Vardenafil"],
        "story": (
            "Pfizer developed sildenafil in the 1990s as a PDE5-inhibiting "
            "vasodilator to treat angina. Trial participants reported an "
            "unexpected side effect, improved erections, which became "
            "the drug's blockbuster indication (Viagra). The same PDE5 "
            "mechanism was later validated for pulmonary arterial "
            "hypertension (Revatio). Tadalafil and vardenafil, the other "
            "major PDE5 inhibitors, share sildenafil's core side-effect "
            "signature (headache, flushing, visual disturbance, priapism) "
            "because they hit the same target."
        ),
    },
    {
        "drug": "Minoxidil",
        "brand": "Loniten / Rogaine",
        "original_use": "Severe, treatment-resistant hypertension",
        "repurposed_use": "Androgenetic alopecia (hair regrowth)",
        "known_relatives": ["Hydralazine"],
        "story": (
            "Minoxidil was developed as a potent oral vasodilator for "
            "severe hypertension. Patients on it grew hair in unexpected "
            "places. Hypertrichosis, a side effect of its vasodilating "
            "mechanism, was then developed into a topical product "
            "for pattern baldness (Rogaine). Hydralazine, another direct "
            "vasodilator antihypertensive, shares much of minoxidil's "
            "cardiovascular side-effect profile (tachycardia, flushing, "
            "edema, headache), reflecting the shared mechanism class."
        ),
    },
    {
        "drug": "Thalidomide",
        "brand": "Thalomid",
        "original_use": "Sedative / antiemetic (withdrawn for teratogenicity)",
        "repurposed_use": "Erythema nodosum leprosum; multiple myeloma",
        "known_relatives": ["Lenalidomide", "Pomalidomide"],
        "story": (
            "Thalidomide's catastrophic teratogenic effects led to its "
            "withdrawal in the 1960s, but its immunomodulatory / "
            "anti-angiogenic mechanism was later found effective for "
            "leprosy complications and, decisively, multiple myeloma. "
            "Lenalidomide and pomalidomide are next-generation analogs "
            "(IMiDs) purpose-built to retain that mechanism and they "
            "carry over the family side-effect signature (cytopenias, "
            "thrombosis risk, teratogenicity) that our fingerprint "
            "approach picks up directly from the data."
        ),
    },
]


st.set_page_config(
    page_title="OffTarget",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #F8FAFC; }
        h1, h2, h3 { color: #1E293B; }
        .sm-card {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .sm-card h4 { margin: 0 0 0.25rem 0; color: #1E293B; }
        .sm-score {
            display: inline-block;
            background: #0D9488;
            color: white;
            border-radius: 999px;
            padding: 0.1rem 0.7rem;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .sm-tag {
            display: inline-block;
            background: #E2E8F0;
            color: #0F172A;
            border-radius: 6px;
            padding: 0.1rem 0.5rem;
            margin: 0.1rem 0.2rem 0 0;
            font-size: 0.8rem;
        }
        .sm-caption { color: #64748B; font-size: 0.85rem; }
        .sm-badge {
            display: inline-block;
            border-radius: 6px;
            padding: 0.1rem 0.55rem;
            margin-top: 0.4rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .sm-badge-shared { background: #DCFCE7; color: #166534; }
        .sm-badge-off-target { background: #FEF3C7; color: #92400E; }
        .sm-badge-unknown { background: #F1F5F9; color: #64748B; font-weight: 400; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_matrix() -> pd.DataFrame:
    return load_matrix()


@st.cache_data
def get_similarity(_matrix: pd.DataFrame, metric: str) -> pd.DataFrame:
    return similarity_matrix(_matrix, metric=metric)


@st.cache_data
def get_targets() -> pd.DataFrame:
    return load_targets()


def target_badge_html(query: str, other: str, targets: pd.DataFrame) -> str:
    relationship = target_relationship(query, other, targets)
    if relationship == "shared":
        family = targets.loc[other, "target_family"]
        return f'<span class="sm-badge sm-badge-shared">Same target: {family}</span>'
    if relationship == "off_target":
        target = targets.loc[other, "primary_target"]
        return f'<span class="sm-badge sm-badge-off-target">Off-target hypothesis: {target}</span>'
    return '<span class="sm-badge sm-badge-unknown">Target unknown</span>'


def render_results_cards(results: pd.DataFrame, query: str, targets: pd.DataFrame) -> None:
    for _, row in results.iterrows():
        tags = "".join(
            f'<span class="sm-tag">{se}</span>'
            for se in row["shared_side_effects"].split(", ")
            if se
        )
        badge = target_badge_html(query, row["drug_name"], targets)
        st.markdown(
            f"""
            <div class="sm-card">
                <h4>{row['drug_name']} <span class="sm-score">{row['similarity']:.1%} match</span></h4>
                <div class="sm-caption">{row['n_shared']} shared side effects</div>
                <div style="margin-top:0.4rem;">{tags}</div>
                <div>{badge}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_score_chart(results: pd.DataFrame, query: str) -> None:
    fig = go.Figure(
        go.Bar(
            x=results["similarity"][::-1],
            y=results["drug_name"][::-1],
            orientation="h",
            marker=dict(color=results["similarity"][::-1], colorscale=CLINICAL_SCALE),
            text=[f"{v:.1%}" for v in results["similarity"][::-1]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Similarity to {query}",
        xaxis_title="Similarity score",
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(300, 40 * len(results)),
        xaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_fingerprint_heatmap(matrix: pd.DataFrame, query: str, results: pd.DataFrame) -> None:
    drugs = [query] + list(results["drug_name"])
    shared_terms: set[str] = set()
    for se_list in results["shared_side_effects"]:
        shared_terms.update(t for t in se_list.split(", ") if t)
    if not shared_terms:
        st.info("No overlapping side effects among the top matches to plot.")
        return
    columns = sorted(shared_terms)
    sub = matrix.loc[drugs, columns]
    fig = go.Figure(
        go.Heatmap(
            z=sub.values,
            x=sub.columns,
            y=sub.index,
            colorscale=[[0, "#F8FAFC"], [1, "#0D9488"]],
            showscale=False,
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title="Side-effect fingerprints (query vs. top matches)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=max(300, 35 * len(drugs)),
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def get_properties() -> pd.DataFrame:
    return pd.read_csv(PROPERTIES_PATH).set_index("drug_name")


@st.cache_resource
def get_viewer_js() -> str:
    """3Dmol.js, vendored locally (BSD-3-Clause, see assets/3Dmol-LICENSE.txt).

    Loaded from disk once and inlined into every viewer's HTML rather than
    fetched from a CDN, so rendering works with no runtime network access.
    """
    return VIEWER_JS_PATH.read_text()


def render_structure_3d(drug: str, height: int = 260) -> None:
    mol_path = STRUCTURES_DIR / f"{drug}.mol"
    if not mol_path.exists():
        st.caption(f"3D structure not available for {drug}.")
        return
    mol_block = json.dumps(mol_path.read_text())
    html = f"""
    <div id="viewer" style="width:{height}px;height:{height}px;"></div>
    <script>{get_viewer_js()}</script>
    <script>
      const viewer = $3Dmol.createViewer(document.getElementById("viewer"), {{backgroundColor: "#F8FAFC"}});
      viewer.addModel({mol_block}, "mol");
      viewer.setStyle({{}}, {{stick: {{colorscheme: "cyanCarbon", radius: 0.15}}}});
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    components.html(html, height=height, width=height)


def render_properties(drug: str, properties: pd.DataFrame) -> None:
    if drug not in properties.index:
        return
    p = properties.loc[drug]
    st.markdown(f"`{p['molecular_formula']}`")
    st.caption(
        f"MW {p['molecular_weight']:.1f} · LogP {p['logp']:.2f}  \n"
        f"H-bond donors/acceptors: {int(p['h_bond_donors'])}/{int(p['h_bond_acceptors'])}  \n"
        f"TPSA {p['tpsa']:.1f} Å² · Rotatable bonds {int(p['rotatable_bonds'])} · "
        f"Rings {int(p['ring_count'])}"
    )


def render_structure_row(drugs: list[str], height: int = 220) -> None:
    properties = get_properties()
    cols = st.columns(len(drugs))
    for col, drug in zip(cols, drugs):
        with col:
            st.markdown(f"**{drug}**")
            render_structure_3d(drug, height=height)
            render_properties(drug, properties)


def search_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Find repurposing candidates by side-effect similarity")
    st.write(
        "Pick a drug and OffTarget will rank every other drug in the "
        "dataset by how similar its side-effect profile is -- regardless "
        "of what disease either drug is actually used for."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        drug = st.selectbox("Drug", options=sorted(matrix.index), index=None,
                             placeholder="Start typing a drug name...")
    with col2:
        metric = st.radio("Similarity metric", options=["jaccard", "cosine"],
                            format_func=str.title)
    with col3:
        n = st.slider("Top N", min_value=3, max_value=20, value=8)

    if not drug:
        st.info("Choose a drug above to see its closest side-effect matches.")
        return

    results = top_n_similar(drug, matrix, n=n, metric=metric)
    if results.empty:
        st.warning("No other drugs share any side effects with this one in the demo dataset.")
        return

    targets = get_targets()
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"#### Top {len(results)} matches for {drug}")
        st.caption(
            "Green = already known to share a target with {}. Amber = an "
            "off-target hypothesis -- high side-effect similarity with no "
            "known shared target, worth investigating.".format(drug)
        )
        render_results_cards(results, drug, targets)
    with right:
        render_score_chart(results, drug)
        render_fingerprint_heatmap(matrix, drug, results)

    with st.expander("View as data table"):
        st.dataframe(results, use_container_width=True, hide_index=True)

    with st.expander("3D structures: query vs. top matches", expanded=False):
        st.caption(
            "Rotate and zoom each structure. Side-effect similarity is a "
            "phenotypic signal, not a chemical one -- these compounds can "
            "(and often do) look nothing alike structurally."
        )
        render_structure_row([drug] + list(results["drug_name"][:4]))


def case_studies_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Does side-effect similarity recover known repurposing discoveries?")
    st.write(
        "Before trusting this method on novel drug pairs, it should pass a "
        "sanity check: for drugs where a side-effect-driven repurposing "
        "story is already documented, does the algorithm actually surface "
        "the known mechanistic relatives near the top of the list?"
    )

    for case in CASE_STUDIES:
        with st.container(border=True):
            st.markdown(f"### {case['drug']} ({case['brand']})")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Originally developed for:** {case['original_use']}")
            c2.markdown(f"**Repurposed for:** {case['repurposed_use']}")
            st.write(case["story"])

            if case["drug"] not in matrix.index:
                st.warning(f"{case['drug']} not in the current dataset.")
                continue

            results = top_n_similar(case["drug"], matrix, n=10, metric="jaccard")
            ranked = list(results["drug_name"])
            hits = [r for r in case["known_relatives"] if r in ranked]

            if hits:
                targets = get_targets()
                for relative in hits:
                    rank = ranked.index(relative) + 1
                    score = results.loc[results["drug_name"] == relative, "similarity"].iloc[0]
                    relationship = target_relationship(case["drug"], relative, targets)
                    if relationship == "shared":
                        target_note = (
                            f" They share a known target "
                            f"({targets.loc[relative, 'target_family']}), confirming the method "
                            f"picked up real biology here."
                        )
                    elif relationship == "off_target":
                        target_note = (
                            f" Interestingly, they have **no known shared target** "
                            f"({targets.loc[case['drug'], 'target_family']} vs. "
                            f"{targets.loc[relative, 'target_family']}) -- exactly the kind of "
                            f"off-target hypothesis this method is meant to surface."
                        )
                    else:
                        target_note = ""
                    st.success(
                        f"Recovered: **{relative}** ranked #{rank} "
                        f"with {score:.1%} Jaccard similarity.{target_note}"
                    )
                render_structure_row([case["drug"]] + hits)
            else:
                st.warning(
                    f"None of {', '.join(case['known_relatives'])} appear in "
                    f"{case['drug']}'s top 10 -- they may be missing from "
                    "the demo dataset."
                )

            with st.expander(f"Full top-10 similarity ranking for {case['drug']}"):
                st.dataframe(results, use_container_width=True, hide_index=True)


def off_target_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Off-target hypotheses")
    st.write(
        "Campillos et al. (*Science*, 2008) proposed side-effect similarity "
        "not just to find repurposing candidates, but to predict shared "
        "molecular **targets** -- including ones nobody had linked a drug "
        "to before. A high-similarity pair that already shares a known "
        "target confirms the method is picking up real biology. A "
        "high-similarity pair with **no known shared target** is a genuine "
        "hypothesis: these two drugs might affect the same underlying "
        "pathway through a mechanism that hasn't been characterized yet."
    )

    min_sim = st.slider(
        "Minimum side-effect similarity (Jaccard)", min_value=0.1, max_value=0.9,
        value=0.25, step=0.05,
    )
    sims = get_similarity(matrix, "jaccard")
    targets = get_targets()
    hypotheses = top_off_target_hypotheses(sims, targets, min_similarity=min_sim, n=15)

    if hypotheses.empty:
        st.info("No off-target hypotheses at this similarity threshold -- try lowering it.")
        return

    st.markdown(f"#### Top {len(hypotheses)} off-target hypotheses in this dataset")
    for _, row in hypotheses.iterrows():
        with st.container(border=True):
            st.markdown(
                f"**{row['drug_a']}** ↔ **{row['drug_b']}** "
                f"<span class='sm-score'>{row['similarity']:.1%} similarity</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"{row['drug_a']}: {row['target_a']}  \n{row['drug_b']}: {row['target_b']}")

    with st.expander("View as data table"):
        st.dataframe(hypotheses, use_container_width=True, hide_index=True)

    st.caption(
        "These are computational leads, not findings -- they'd need "
        "experimental validation (e.g. binding assays) before meaning "
        "anything clinically, exactly as in the original Campillos et al. "
        "study."
    )


def about_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        **Concept.** Drugs that cause similar patterns of side effects often
        act on the same underlying biology, even when they're prescribed for
        completely different diseases. Side-effect similarity can therefore
        surface drug repurposing candidates that pure indication-based
        search would never suggest.

        **Why this matters.** Repurposing an already-approved drug skips
        most of the cost of drug development -- its safety profile, dosing,
        and manufacturing are already established, so a new indication can
        go almost straight to efficacy trials. Traditional repurposing
        search starts from a known mechanism or indication, which means it
        can only find what's already understood. Side-effect similarity
        needs no prior knowledge of mechanism: it surfaces Tadalafil and
        Vardenafil as Sildenafil's relatives purely from overlapping side
        effects, without ever being told "these are all PDE5 inhibitors."
        The tradeoff is that it's a hypothesis generator, not a validator --
        a high score means "worth investigating," not "will work," which is
        why the Validated Case Studies tab exists: to confirm the method
        recovers *known* good leads before trusting it on unknown ones.

        **Fingerprints.** Each drug is represented as a binary vector over a
        vocabulary of side-effect terms: 1 if the drug is documented to
        cause that effect, 0 otherwise. This mirrors how the
        [SIDER database](http://sideeffects.embl.de/) (Side Effect Resource)
        encodes drug/adverse-effect relationships from public drug labels.

        **Similarity metrics.**
        - *Jaccard index*: `|A ∩ B| / |A ∪ B|` -- the fraction of all side
          effects (across both drugs) that they share. Penalizes drugs with
          very different total side-effect counts.
        - *Cosine similarity*: the cosine of the angle between the two
          binary vectors. Slightly more forgiving of drugs with many
          reported effects.

        **Data source.** This deployment ships with a curated, hand-built
        demo dataset (61 drugs, 96 MedDRA-style side-effect terms) covering
        diverse drug classes and every case study on the previous tab,
        because this environment can't reach sideeffects.embl.de directly to
        download the full SIDER database. `data_prep.py` will automatically
        use the real SIDER TSVs instead if you download `meddra_all_se.tsv.gz`
        and `drug_names.tsv` from SIDER and place them in `data/raw/` --
        no code changes required.

        **Off-target hypotheses.** Alongside each search result and in its
        own tab, OffTarget shows whether a similar drug shares a curated,
        known molecular target with the query drug (confirming the method)
        or has no known shared target despite high side-effect similarity
        (an off-target hypothesis worth investigating). This framing
        follows Campillos et al., "Drug target identification using
        side-effect similarity" (*Science*, 2008) -- the paper this whole
        approach is built on, which used exactly this logic to predict and
        experimentally validate several previously-unknown drug targets.

        **3D structures.** Each drug's structure is generated offline from a
        curated SMILES string with RDKit, validated against its expected
        molecular formula, and rendered with a locally vendored copy of
        3Dmol.js -- no CDN or live structure-database lookup involved.
        Molecular weight, LogP, H-bond donor/acceptor counts, TPSA,
        rotatable bond count, and ring count are computed the same way and
        shown alongside each structure. See the README for the full
        pipeline and its accuracy caveats.

        **Limitations.** The demo dataset is illustrative, not exhaustive --
        absence of a shared side effect here means it wasn't included in
        this curated list, not that it doesn't exist. Side-effect
        co-occurrence is also a weak proxy for shared mechanism: it can
        reflect genuine target overlap, but also coincidence, drug-class
        conventions in labeling, or reporting bias. Treat results as
        hypothesis-generating leads, not conclusions.
        """
    )
    st.markdown(f"**Current dataset:** {matrix.shape[0]} drugs x {matrix.shape[1]} side effects.")


def main() -> None:
    inject_css()
    st.title("OffTarget")
    st.caption(
        "Drug repurposing candidates via side-effect similarity, in the "
        "spirit of the SIDER database."
    )

    matrix = get_matrix()

    with st.sidebar:
        st.markdown("### About OffTarget")
        st.write(
            "Search a drug to find others with the most similar reported "
            "side-effect profile -- potential repurposing leads driven by "
            "shared biology, not shared indication."
        )
        st.markdown("---")
        st.markdown(
            f"**Dataset:** {matrix.shape[0]} drugs, {matrix.shape[1]} side effects"
        )
        st.markdown(
            "Built on a SIDER-style methodology. "
            "[Source](https://github.com/vstimpson/OffTarget)"
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Search", "Validated Case Studies", "Off-Target Hypotheses", "Methodology"]
    )
    with tab1:
        search_tab(matrix)
    with tab2:
        case_studies_tab(matrix)
    with tab3:
        off_target_tab(matrix)
    with tab4:
        about_tab(matrix)


if __name__ == "__main__":
    main()
