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

from analysis import (
    cluster_category_purity,
    cluster_drugs,
    load_categories,
    pca_projection,
    surprising_pairs,
    tsne_projection,
)
from similarity import explain_similarity, idf_weights, load_matrix, similarity_matrix, top_n_similar
from targets import (
    all_pairs_target_overlap,
    drug_reframing_signals,
    load_reframings,
    load_targets,
    reframing_candidates,
    target_overlap_by_similarity_bin,
    target_overlap_correlation,
    target_relationship,
    top_off_target_hypotheses,
)

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
def get_similarity(_matrix: pd.DataFrame, metric: str, weighted: bool = False) -> pd.DataFrame:
    return similarity_matrix(_matrix, metric=metric, weighted=weighted)


@st.cache_data
def get_idf_weights(_matrix: pd.DataFrame) -> pd.Series:
    return idf_weights(_matrix)


@st.cache_data
def get_targets() -> pd.DataFrame:
    return load_targets()


@st.cache_data
def get_reframings() -> pd.DataFrame:
    return load_reframings()


@st.cache_data
def get_categories() -> pd.DataFrame:
    return load_categories()


@st.cache_data
def get_pca(_matrix: pd.DataFrame) -> pd.DataFrame:
    return pca_projection(_matrix)


@st.cache_data
def get_tsne(_matrix: pd.DataFrame) -> pd.DataFrame:
    return tsne_projection(_matrix)


@st.cache_data
def get_clusters(_matrix: pd.DataFrame, n_clusters: int, method: str) -> pd.Series:
    return cluster_drugs(_matrix, n_clusters=n_clusters, method=method)


def target_badge_html(query: str, other: str, targets: pd.DataFrame) -> str:
    relationship = target_relationship(query, other, targets)
    if relationship == "shared":
        family = targets.loc[other, "target_family"]
        return f'<span class="sm-badge sm-badge-shared">Same target: {family}</span>'
    if relationship == "off_target":
        target = targets.loc[other, "primary_target"]
        return f'<span class="sm-badge sm-badge-off-target">Off-target hypothesis: {target}</span>'
    return '<span class="sm-badge sm-badge-unknown">Target unknown</span>'


def render_results_cards(
    results: pd.DataFrame,
    query: str,
    targets: pd.DataFrame,
    matrix: pd.DataFrame | None = None,
    categories: pd.DataFrame | None = None,
    weighted: bool = True,
) -> None:
    for _, row in results.iterrows():
        other = row["drug_name"]
        tags = "".join(
            f'<span class="sm-tag">{se}</span>'
            for se in row["shared_side_effects"].split(", ")
            if se
        )
        badge = target_badge_html(query, other, targets)
        category_line = ""
        if categories is not None and other in categories.index:
            category_line = f'<div class="sm-caption">{categories.loc[other, "therapeutic_category"]}</div>'

        explanation_line = ""
        if matrix is not None:
            explanation = explain_similarity(query, other, matrix, weighted=weighted, top_k=3)
            if not explanation.empty:
                terms = ", ".join(explanation["side_effect"])
                explanation_line = f'<div class="sm-caption">Driven mostly by: {terms}</div>'

        st.markdown(
            f"""
            <div class="sm-card">
                <h4>{other} <span class="sm-score">{row['similarity']:.1%} match</span></h4>
                {category_line}
                <div class="sm-caption">{row['n_shared']} shared side effects</div>
                <div style="margin-top:0.4rem;">{tags}</div>
                {explanation_line}
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


def render_reframing_signals(drug: str, matrix: pd.DataFrame) -> None:
    """Flag side effects of `drug` with real precedent for becoming the
    actual therapeutic purpose -- the Viagra/Rogaine pattern generalized.
    """
    reframings = get_reframings()
    signals = drug_reframing_signals(drug, matrix, reframings)
    if signals.empty:
        return

    for _, row in signals.iterrows():
        if row["is_pioneer"]:
            st.success(
                f"**{drug}** is the textbook example: its **{row['side_effect']}** "
                f"side effect became **{row['reframed_purpose']}**. {row['note']}"
            )
        else:
            st.info(
                f"**{drug}** also causes **{row['side_effect']}**, the same side "
                f"effect that became **{row['reframed_purpose']}** for "
                f"**{row['pioneer_drug']}**. By that precedent, {drug} could be "
                f"a candidate worth investigating for the same purpose."
            )


def search_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Find repurposing candidates by side-effect similarity")
    st.write(
        "Pick a drug and OffTarget will rank every other drug in the "
        "dataset by how similar its side-effect profile is -- regardless "
        "of what disease either drug is actually used for."
    )

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1.2])
    with col1:
        drug = st.selectbox("Drug", options=sorted(matrix.index), index=None,
                             placeholder="Start typing a drug name...")
    with col2:
        metric = st.radio(
            "Similarity metric", options=["jaccard", "cosine"], format_func=str.title,
            help="Two ways to score how much two checklists overlap. "
                 "Jaccard: shared side effects divided by all side effects "
                 "either drug has. Cosine: a similar idea, just more "
                 "forgiving toward drugs with a lot of reported side "
                 "effects. They usually agree; the exact number rarely "
                 "matters as much as the ranking.",
        )
    with col3:
        n = st.slider("Top N", min_value=3, max_value=20, value=8)
    with col4:
        weighted = st.checkbox(
            "IDF-weighted", value=True,
            help="Down-weight common side effects (headache, nausea) and "
                 "up-weight rare ones, the way TF-IDF weights words. "
                 "Measurably improves agreement with known drug targets "
                 "(0.41 -> 0.51 correlation) -- see the Validated Case "
                 "Studies tab.",
        )

    if not drug:
        st.info("Choose a drug above to see its closest side-effect matches.")
        return

    categories = get_categories()
    targets = get_targets()
    category = categories.loc[drug, "therapeutic_category"] if drug in categories.index else "unknown"
    target = targets.loc[drug, "primary_target"] if drug in targets.index else "unknown"
    n_effects = int(matrix.loc[drug].sum())
    st.markdown(
        f"**{drug}** &nbsp;·&nbsp; {category} &nbsp;·&nbsp; target: {target} "
        f"&nbsp;·&nbsp; {n_effects} documented side effects"
    )

    render_reframing_signals(drug, matrix)

    results = top_n_similar(drug, matrix, n=n, metric=metric, weighted=weighted)
    if results.empty:
        st.warning("No other drugs share any side effects with this one in the demo dataset.")
        return

    left, right = st.columns([1, 1])
    with left:
        weight_note = " (IDF-weighted)" if weighted else " (unweighted)"
        st.markdown(f"#### Top {len(results)} matches for {drug}{weight_note}")
        st.caption(
            "Green = already known to share a target with {}. Amber = an "
            "off-target hypothesis -- high side-effect similarity with no "
            "known shared target, worth investigating.".format(drug)
        )
        render_results_cards(results, drug, targets, matrix=matrix, categories=categories, weighted=weighted)
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

    st.markdown("---")
    render_target_validation(matrix)


def render_target_validation(matrix: pd.DataFrame) -> None:
    st.markdown("#### Does this hold up across the whole dataset?")
    st.write(
        "The three case studies above are hand-picked. A more honest test: "
        "take *every* drug pair with a curated target, and check whether "
        "higher side-effect similarity actually corresponds to a higher "
        "rate of sharing a known target -- not just for the drugs the "
        "case studies were built around."
    )

    targets = get_targets()
    pairs_unweighted = all_pairs_target_overlap(get_similarity(matrix, "jaccard", False), targets)
    pairs_weighted = all_pairs_target_overlap(get_similarity(matrix, "jaccard", True), targets)
    stats_unweighted = target_overlap_correlation(pairs_unweighted)
    stats_weighted = target_overlap_correlation(pairs_weighted)

    if pairs_unweighted.empty:
        st.info("Not enough curated-target pairs to validate.")
        return

    st.markdown("###### Does IDF weighting actually help?")
    st.write(
        "Down-weighting common side effects (headache, nausea) and "
        "up-weighting rare ones is a specific, testable claim: it should "
        "make similarity track known targets *more* closely, not less. "
        "Same dataset, same pairs, same target curation only the "
        "weighting changes:"
    )
    wc1, wc2 = st.columns(2)
    wc1.metric(
        "Unweighted Jaccard correlation", f"{stats_unweighted['correlation']:.3f}",
    )
    wc2.metric(
        "IDF-weighted Jaccard correlation", f"{stats_weighted['correlation']:.3f}",
        delta=f"{stats_weighted['correlation'] - stats_unweighted['correlation']:+.3f}",
    )

    weighted_view = st.toggle("Show IDF-weighted results below", value=True)
    pairs = pairs_weighted if weighted_view else pairs_unweighted
    stats = stats_weighted if weighted_view else stats_unweighted

    c1, c2, c3 = st.columns(3)
    c1.metric("Pairs analyzed", f"{stats['n_pairs']:,}")
    c2.metric("Correlation (similarity vs. shared target)", f"{stats['correlation']:.2f}")
    c3.metric(
        "Mean similarity, shared vs. not",
        f"{stats['mean_sim_shared']:.1%} vs. {stats['mean_sim_not_shared']:.1%}",
    )

    bins = target_overlap_by_similarity_bin(pairs, [0, 0.1, 0.2, 0.3, 0.5, 1.0])
    fig = go.Figure(
        go.Bar(
            x=bins["bin_label"],
            y=bins["pct_shared_target"] * 100,
            marker=dict(color=bins["pct_shared_target"], colorscale=CLINICAL_SCALE),
            text=[f"n={n}" for n in bins["n_pairs"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Share of drug pairs with a known shared target, by similarity range",
        xaxis_title="Jaccard similarity range",
        yaxis_title="% sharing a known target",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"A correlation of {stats['correlation']:.2f} across {stats['n_pairs']:,} pairs, "
        "and a shared-target rate that climbs with similarity, is the general-case "
        "version of what the case studies show individually: this isn't just "
        "picking three examples that happen to work. It's also not proof the "
        "method is reliable for any single pair, most high-similarity pairs "
        "*still* don't share a known target, which is exactly the off-target "
        "hypothesis space the previous tab explores."
    )


def off_target_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Off-target hypotheses")
    st.write(
        "When two drugs share a lot of side effects, there are two "
        "possibilities: either scientists already know why (they act on "
        "the same target in the body), or nobody has confirmed a reason "
        "yet. That second case is the interesting one: it's a hint that "
        "two drugs might work through a related mechanism that hasn't "
        "been pinned down. This idea comes from a real study (Campillos et "
        "al., *Science*, 2008), which used exactly this logic to predict "
        "molecular targets nobody had linked to a given drug before. Two "
        "views of it live on this tab: which drug *pairs* look related "
        "with no confirmed reason yet, and further down, which specific "
        "*side effects* have real precedent for becoming a drug's actual "
        "purpose."
    )

    sl_col, cb_col = st.columns([3, 1])
    with sl_col:
        min_sim = st.slider(
            "Minimum side-effect similarity (Jaccard)", min_value=0.1, max_value=0.9,
            value=0.25, step=0.05,
            help="How much side-effect checklist overlap a pair needs before "
                 "it counts. Higher means fewer, more strongly matched pairs.",
        )
    with cb_col:
        weighted = st.checkbox("IDF-weighted", value=True, key="offtarget_weighted")
    sims = get_similarity(matrix, "jaccard", weighted)
    targets = get_targets()
    hypotheses = top_off_target_hypotheses(sims, targets, min_similarity=min_sim, n=15)

    if hypotheses.empty:
        st.info("No off-target hypotheses at this similarity threshold -- try lowering it.")
    else:
        st.markdown(f"#### Top {len(hypotheses)} off-target hypotheses in this dataset")
        for _, row in hypotheses.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{row['drug_a']}** ↔ **{row['drug_b']}** "
                    f"<span class='sm-score'>{row['similarity']:.1%} similarity</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{row['drug_a']}: {row['target_a']}  \n{row['drug_b']}: {row['target_b']}"
                )
                with st.expander("3D structures and properties"):
                    render_structure_row([row["drug_a"], row["drug_b"]])

        with st.expander("View as data table"):
            st.dataframe(hypotheses, use_container_width=True, hide_index=True)

        st.caption(
            "These are computational leads, not findings. They'd need "
            "experimental validation (e.g. binding assays) before meaning "
            "anything clinically, exactly as in the original Campillos et al. "
            "study."
        )

    st.markdown("---")
    st.markdown("#### Reframed side effects: known precedent")
    st.write(
        "The most direct version of this idea, generalized beyond three "
        "hardcoded stories: sildenafil's priapism became Viagra; "
        "minoxidil's hypertrichosis became Rogaine. Below, every other drug "
        "in the dataset that shares one of these precedent side effects, "
        "an untapped candidate for the same reframed purpose, by the same "
        "logic."
    )
    reframings = get_reframings()
    candidates = reframing_candidates(matrix, reframings)
    if candidates.empty:
        st.info("No reframing candidates found in the current dataset.")
    else:
        st.dataframe(candidates, use_container_width=True, hide_index=True)


def surprising_pairs_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Surprising pairs")
    st.write(
        "The investigation this whole app is built around, run directly: "
        "which drug pairs are highly similar by side effects despite "
        "treating completely different conditions? A drug for epilepsy and "
        "a drug for hypertension showing 0.87 cosine similarity is exactly "
        "the kind of lead indication-based search would never surface."
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        min_sim = st.slider(
            "Minimum similarity", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
            key="sp_min_sim",
            help="How much side-effect checklist overlap a pair needs "
                 "before it shows up below. Higher means fewer, more "
                 "strongly matched pairs.",
        )
    with c2:
        metric = st.radio(
            "Metric", options=["cosine", "jaccard"], format_func=str.title, key="sp_metric",
            help="Two ways to score how much two checklists overlap. "
                 "Jaccard: shared side effects divided by all side effects "
                 "either drug has. Cosine: a similar idea, just more "
                 "forgiving toward drugs with a lot of reported side "
                 "effects.",
        )
    with c3:
        sort_by = st.radio(
            "Sort by", options=["similarity", "repurposing_score"],
            format_func=lambda s: "Similarity" if s == "similarity" else "Repurposing score",
            key="sp_sort",
        )

    weighted = st.checkbox("IDF-weighted", value=True, key="sp_weighted")

    sims = get_similarity(matrix, metric, weighted)
    categories = get_categories()
    targets = get_targets()
    pairs = surprising_pairs(sims, categories, targets, min_similarity=min_sim, n=20, sort_by=sort_by)

    if pairs.empty:
        st.info("No cross-category pairs at this similarity threshold -- try lowering it.")
        return

    st.markdown(f"#### Top {len(pairs)} surprising pairs")
    for _, row in pairs.iterrows():
        with st.container(border=True):
            badge_class = {
                "shared": "sm-badge-shared", "off_target": "sm-badge-off-target",
                "unknown": "sm-badge-unknown",
            }[row["target_relationship"]]
            badge_text = {
                "shared": "Known shared target", "off_target": "No known shared target",
                "unknown": "Target unknown",
            }[row["target_relationship"]]
            st.markdown(
                f"**{row['drug_a']}** ({row['category_a']}) ↔ "
                f"**{row['drug_b']}** ({row['category_b']})  \n"
                f"<span class='sm-score'>{row['similarity']:.1%} similarity</span> "
                f"<span class='sm-badge {badge_class}'>{badge_text}</span> "
                f"<span class='sm-caption'>repurposing score: {row['repurposing_score']:.3f}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"{row['drug_a']}: {row['target_a']}  \n{row['drug_b']}: {row['target_b']}")

            explanation = explain_similarity(row["drug_a"], row["drug_b"], matrix, weighted=weighted)
            if not explanation.empty:
                terms = ", ".join(
                    f"{r['side_effect']} ({r['pct_of_shared_weight']:.0f}%)"
                    for _, r in explanation.iterrows()
                )
                st.caption(f"Strongest contributing shared side effects: {terms}")

            with st.expander("3D structures and properties"):
                render_structure_row([row["drug_a"], row["drug_b"]])

    with st.expander("View as data table"):
        st.dataframe(pairs, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### The repurposing score")
    st.latex(
        r"\text{Repurposing Score} = \text{Side-effect similarity} "
        r"\times \text{Biological plausibility} \times \text{Indication difference}"
    )
    st.write(
        "An **exploratory** ranking, not a validated clinical metric. "
        "Biological plausibility is a stand-in built from curated target "
        "data: 1.0 if the pair already shares a known target, 0.5 if "
        "there's no known shared target (plausible, unconfirmed), 0.3 if "
        "either drug's target isn't curated at all. Indication difference "
        "is 1.0 for a genuine cross-category pair (all pairs on this page) "
        "and would be 0.2 for same-category pairs, which this page filters "
        "out entirely since they're not surprising. The formula is "
        "deliberately simple -- it exists to rank leads for further "
        "investigation, not to make a scientific claim about any one pair."
    )


def cluster_map_tab(matrix: pd.DataFrame) -> None:
    st.subheader("Cluster map")
    st.write(
        "Every drug here is really just a checklist of up to 96 possible "
        "side effects (see the glossary in the sidebar if any term on this "
        "page is unfamiliar). This tab draws those checklists as a picture "
        "you can look at, and checks whether drugs that end up looking "
        "alike by their checklist alone also happen to be the same kind of "
        "drug in real life."
    )

    with st.expander("What's the difference between PCA and t-SNE?"):
        st.markdown(
            """
**PCA (Principal Component Analysis)** is a linear projection. It finds
the two directions through the 96-item side-effect checklist that
capture the most spread (variance) across all 61 drugs, and plots each
drug's position along those two directions. Because it's linear and
deterministic, the same drug always lands in the same place, and the axes
have a real, if abstract, meaning: "direction of most variation," "second
most variation." Its weakness: if the interesting structure in the data is
a tight, non-linear clustering rather than broad spread, PCA can wash it
out. Two genuinely similar drugs can end up looking far apart if the
dominant variance in the dataset runs a different direction.

**t-SNE (t-distributed Stochastic Neighbor Embedding)** is non-linear, and
optimized for a different goal: keep points that are close neighbors in
the original 96D space close together in the 2D plot, without caring
whether it preserves distances between far-apart points or the overall
shape. This is why it tends to produce visually tighter, more dramatic
clusters than PCA, since it's built specifically to reveal local grouping.
The tradeoffs: it's stochastic (a different random seed can shuffle the
layout, though the seed is fixed here for reproducibility), sensitive to a
tuning parameter (perplexity), and, importantly, you cannot compare the
distance between two far-apart clusters and conclude anything from it.
Only "near" versus "not near" is meaningful.

**Why both are offered side by side:** they fail in different ways, so
agreement between them is more convincing than either alone. If two drugs
land close together under both PCA (a global, linear method) and t-SNE (a
local, non-linear method), that's a stronger signal than either method
individually, since it isn't just an artifact of one algorithm's
particular bias. Where they disagree is also informative: it flags cases
where "closeness" depends on which notion of similarity is used, which is
worth being upfront about rather than picking one method and presenting
it as definitive.
            """
        )

    method = st.radio("Projection", options=["PCA", "t-SNE"], horizontal=True)
    color_by = st.radio(
        "Color by", options=["therapeutic_category", "target_family"],
        format_func=lambda s: "Therapeutic category" if s == "therapeutic_category" else "Known target family",
        horizontal=True,
    )

    coords = get_pca(matrix) if method == "PCA" else get_tsne(matrix)
    categories = get_categories()
    targets = get_targets()
    color_source = categories["therapeutic_category"] if color_by == "therapeutic_category" else targets["target_family"]

    plot_df = coords.join(color_source.rename("group"), how="left")
    plot_df["group"] = plot_df["group"].fillna("unknown")

    fig = go.Figure()
    for group, sub in plot_df.groupby("group"):
        fig.add_trace(
            go.Scatter(
                x=sub["x"], y=sub["y"], mode="markers+text",
                text=sub.index, textposition="top center",
                textfont=dict(size=8),
                name=group[:40],
                marker=dict(size=9),
            )
        )
    if method == "PCA":
        axis_title = "Principal component {} (arbitrary units -- only relative distance matters)"
    else:
        axis_title = "t-SNE dimension {} (no fixed meaning -- only nearby points are comparable)"
    fig.update_layout(
        title=f"{method} projection of side-effect fingerprints",
        xaxis_title=axis_title.format(1),
        yaxis_title=axis_title.format(2),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=40),
        height=650,
        legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each point is one drug, positioned using its full 96-item "
        "side-effect checklist. Neither axis corresponds to a specific "
        "side effect or has physical units. PCA axes are directions of "
        "greatest variance across all checklists combined; t-SNE axes "
        "preserve which drugs are near neighbors, not true distances. What "
        "matters is relative position: same-colored points landing close "
        "together means side-effect data alone reconstructed a real drug "
        "class, with no indication or target info given to the projection."
    )

    st.markdown("---")
    st.markdown("#### Clustering: does grouping by side effects alone recover drug classes?")
    st.write(
        "We group drugs together based only on how similar their side "
        "effect patterns are, using clustering methods such as K-means or "
        "hierarchical clustering. The algorithm is never told what type of "
        "drug each one is. After the clusters are formed, we compare them "
        "with the drugs' real therapeutic classes. If most drugs in a "
        "cluster belong to the same known drug class, that cluster has "
        "high purity, which suggests that side-effect patterns alone "
        "contain enough information to recover meaningful similarities in "
        "how drugs work."
    )
    st.write(
        "K-means clustering makes you choose the number of groups, K, in "
        "advance. For example, with 100 drugs and K = 5, the algorithm "
        "tries to divide those drugs into five groups. Hierarchical "
        "clustering instead builds a family tree of drugs: it finds the "
        "two most similar drugs and joins them, then finds the next "
        "closest drug or group, and so on until everything is connected. "
        "The table below shows the resulting cluster each drug ended up "
        "in either way, not the full tree structure."
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        cluster_method = st.radio(
            "Clustering method", options=["kmeans", "hierarchical"],
            format_func=lambda s: "K-means" if s == "kmeans" else "Hierarchical",
            horizontal=True,
        )
    with cc2:
        n_clusters = st.slider("Number of clusters", min_value=4, max_value=20, value=10)

    clusters = get_clusters(matrix, n_clusters, cluster_method)
    purity = cluster_category_purity(clusters, categories)
    st.metric("Mean cluster purity", f"{purity['purity'].mean():.1%}")
    st.dataframe(purity, use_container_width=True, hide_index=True)
    st.caption(
        "Purity of 1.0 means every drug in that cluster shares the same "
        "therapeutic category. Purity well below 1.0 for most clusters is "
        "expected and informative, not a failure: it's the same story as "
        "the off-target hypotheses tab -- drugs from different classes "
        "landing in the same side-effect cluster are exactly the surprising "
        "pairs worth a closer look, not noise to explain away."
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

        **IDF weighting.** Headache and nausea appear in most drugs; a rare
        effect like angioedema appears in almost none. Counting every side
        effect equally treats them as equally informative, which they
        aren't. OffTarget can weight each side effect by
        `w = log(N / n)` (N = total drugs, n = drugs with that side effect)
        -- the same idea as IDF in TF-IDF, applied to a presence matrix
        instead of word counts. A side effect present in every drug gets
        weight 0; a side effect present in one drug out of sixty gets the
        highest weight. This isn't just a plausible tweak: on this dataset
        it measurably improves agreement with known drug targets (Jaccard
        correlation with shared-target status rises from 0.41 unweighted to
        0.51 weighted -- see the Validated Case Studies tab for the live
        comparison). Toggle it on the Search and Off-Target Hypotheses tabs.
        """
    )
    weights = get_idf_weights(matrix)
    wc1, wc2 = st.columns(2)
    with wc1:
        st.caption("Least informative (most common) side effects")
        st.dataframe(
            weights.sort_values().head(5).rename("weight").reset_index(),
            hide_index=True, use_container_width=True,
        )
    with wc2:
        st.caption("Most informative (rarest) side effects")
        st.dataframe(
            weights.sort_values(ascending=False).head(5).rename("weight").reset_index(),
            hide_index=True, use_container_width=True,
        )
    st.markdown(
        """
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
        The **Validated Case Studies** tab also runs this check across the
        *whole* dataset, not just three examples: does higher side-effect
        similarity actually correspond to a higher rate of sharing a known
        target? (Short answer: yes -- see that tab for the numbers.)

        **Reframed side effects.** A more literal reading of "bad side
        effects, used positively": some side effects have real precedent
        for becoming a drug's actual purpose (sildenafil's priapism became
        Viagra; minoxidil's hypertrichosis became Rogaine; topiramate's and
        bupropion's weight-loss side effect became Qsymia and Contrave).
        OffTarget generalizes each precedent to every other drug that
        shares that side effect, flagging it as an untapped candidate for
        the same reframed purpose -- shown for the searched drug on the
        Search tab, and across the whole dataset on the Off-Target
        Hypotheses tab.

        **Surprising pairs.** The Search and Off-Target Hypotheses tabs work
        one query drug at a time. The Surprising Pairs tab runs the same
        idea as a dataset-wide scan: every drug pair above a similarity
        threshold that belongs to *different curated therapeutic
        categories* (`data/raw/drug_categories.csv`), with an explanation
        layer showing exactly which shared side effects drive each score --
        not just "0.82 similar," but "12 shared side effects, led by
        neuropathy and dry mouth." Each pair also gets an exploratory
        **repurposing score** (similarity x biological plausibility x
        indication difference) for ranking leads; see that tab for the
        formula and its explicit caveats.

        **Cluster map.** A PCA or t-SNE projection turns each drug's 96-item
        side-effect checklist into a single dot on a 2D picture, colored by
        therapeutic category or target family -- if same-class drugs cluster
        together visually, that's evidence the checklists encode real
        pharmacology. The same tab runs K-means or hierarchical clustering
        on the checklists alone (categories are never given to the
        algorithm) and measures cluster purity against the real drug
        classes -- a quantitative, not just visual, version of the same
        question.

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
        with st.expander("New here? Plain-language glossary"):
            st.markdown(
                """
**Side-effect checklist (the "fingerprint").** Picture a list of every
side effect in the dataset, 96 of them. Each drug gets a checkmark next
to every side effect it's known to cause, and an empty box for every one
it doesn't. That checklist is all this app actually knows about a drug.

**Similarity score.** How much two drugs' checklists overlap. If two
drugs both check off "dizziness," "flushing," and "headache" and little
else, they score high, even if one treats heart disease and the other
treats depression.

**Target.** The specific protein in the body a drug physically locks
onto, like a lock and key: a receptor, an enzyme, a channel. Two drugs
can look completely different chemically and still act on the same
target.

**Off-target hypothesis.** Two drugs whose checklists overlap a lot, but
where nobody has confirmed they share a target. Worth investigating, not
a proven finding.

**The map (PCA / t-SNE).** Two different ways of drawing all the drugs'
checklists as dots on a single page, positioned so similar checklists
land near each other. Think of it like a seating chart based purely on
who has the same interests, not a physical map with real distances.

**Clustering.** Automatically sorting drugs into groups based only on
their checklists, then checking whether those groups happen to match
real, known drug classes.

**Repurposing score.** An exploratory ranking that combines the
similarity score with how plausible a shared mechanism is and how
different the two drugs' current uses are, meant for prioritizing leads,
not as a scientific verdict.
                """
            )
        st.markdown("---")
        st.markdown(
            f"**Dataset:** {matrix.shape[0]} drugs, {matrix.shape[1]} side effects"
        )
        st.markdown(
            "Built on a SIDER-style methodology. "
            "[Source](https://github.com/vstimpson/OffTarget)"
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Search", "Validated Case Studies", "Off-Target Hypotheses",
            "Surprising Pairs", "Cluster Map", "Methodology",
        ]
    )
    with tab1:
        search_tab(matrix)
    with tab2:
        case_studies_tab(matrix)
    with tab3:
        off_target_tab(matrix)
    with tab4:
        surprising_pairs_tab(matrix)
    with tab5:
        cluster_map_tab(matrix)
    with tab6:
        about_tab(matrix)


if __name__ == "__main__":
    main()
