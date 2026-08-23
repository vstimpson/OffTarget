"""Structured biological pathway data for OffTarget.

The existing target_family column in drug_targets.csv is a flat label
("Phosphodiesterase type 5 (PDE5)"). That's enough to say two drugs hit the
*same* target, but it can't say anything about two drugs that hit
*different* targets sitting on the *same* underlying pathway -- for
example an ACE inhibitor and an AT1 receptor blocker, which act at
different steps of the same blood-pressure pathway, or Sildenafil (blocks
PDE5) and Minoxidil (opens a potassium channel), which act through
different molecules but both end in the same result: vascular smooth
muscle relaxation.

This module models each pathway as one or more ordered chains ("branches")
of nodes (biological states or molecules) connected by edges (the
enzyme/receptor/process that converts one state into the next). A single
pathway can have several branches that start differently but converge on
the same downstream effect -- that convergence is exactly what makes two
drugs with unrelated targets show up as similar by side effects.

TARGET_TO_PATHWAY then maps each curated target_family (see
data/raw/drug_targets.csv) onto the specific pathway/branch/edge it acts
on, plus a short verb describing what the drug does there. A target family
can map to more than one entry (e.g. an SNRI acts on two branches at once).

This is a simplified, educational model, not a clinical or pharmacological
reference -- see the Methodology tab for that caveat in context.
"""
from __future__ import annotations

PATHWAYS: dict[str, dict] = {
    "raas": {
        "name": "Blood pressure control pathway (RAAS)",
        "summary": "The renin-angiotensin-aldosterone system: the body's main lever for raising blood pressure when it senses low blood volume or salt.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Angiotensinogen (an inactive protein made by the liver)",
                    "Angiotensin I (still inactive)",
                    "Angiotensin II (the active signal)",
                    "Blood vessels narrow and the kidneys retain salt: blood pressure rises",
                ],
                "edges": [
                    "Renin cuts it into",
                    "ACE converts it into",
                    "Binds the AT1 receptor, triggering",
                ],
            }
        ],
    },
    "vascular_relaxation": {
        "name": "Vascular smooth muscle relaxation pathway",
        "summary": "Several unrelated molecular starting points all end at the same place: the muscle around blood vessels relaxes and the vessel widens. This is why drugs with completely different targets can still cause similar side effects (flushing, headache, low blood pressure).",
        "branches": [
            {
                "label": "Nitric oxide / cGMP branch",
                "nodes": [
                    "Nitric oxide (NO) signal",
                    "cGMP level in the muscle cell",
                    "Vascular smooth muscle relaxation",
                ],
                "edges": [
                    "Activates guanylate cyclase, producing",
                    "PDE5 normally breaks cGMP back down; blocking PDE5 keeps it high, sustaining",
                ],
            },
            {
                "label": "Potassium channel branch",
                "nodes": [
                    "Smooth muscle cell membrane potential",
                    "Channel opens, cell hyperpolarizes (calms down)",
                    "Vascular smooth muscle relaxation",
                ],
                "edges": [
                    "KATP channel opens, causing",
                    "A calmer cell produces",
                ],
            },
            {
                "label": "Calcium channel branch",
                "nodes": [
                    "Calcium outside the cell",
                    "Calcium inside the cell (normally triggers contraction)",
                    "Vascular smooth muscle relaxation",
                ],
                "edges": [
                    "L-type calcium channel normally lets it in; blocking the channel reduces",
                    "Less calcium means less contraction, i.e.",
                ],
            },
            {
                "label": "Direct branch",
                "nodes": [
                    "Smooth muscle cell",
                    "Vascular smooth muscle relaxation",
                ],
                "edges": [
                    "Direct vasodilator action (the exact molecular mechanism is not fully resolved)",
                ],
            },
        ],
    },
    "cholesterol": {
        "name": "Cholesterol synthesis pathway",
        "summary": "The liver's assembly line for building cholesterol from scratch.",
        "branches": [
            {
                "label": None,
                "nodes": ["Acetyl-CoA (basic building block)", "Mevalonate", "Cholesterol"],
                "edges": [
                    "HMG-CoA reductase converts it (the rate-limiting step)",
                    "Several more enzyme steps build up to",
                ],
            }
        ],
    },
    "coagulation": {
        "name": "Blood clotting pathway",
        "summary": "How the body activates clotting factors to stop bleeding.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Vitamin K (used up after each clotting reaction)",
                    "Recycled, active vitamin K",
                    "Active clotting factors II, VII, IX, X",
                    "Blood clot forms",
                ],
                "edges": [
                    "VKORC1 recycles it back into",
                    "Vitamin K activates",
                    "which trigger",
                ],
            }
        ],
    },
    "platelet_activation": {
        "name": "Platelet activation pathway",
        "summary": "How platelets (the cell fragments that plug injured blood vessels) get switched on. A separate mechanism from the clotting-factor cascade above.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Blood vessel injury",
                    "ADP released, binds platelet receptors",
                    "Platelet fully activates and changes shape",
                    "Platelets clump together: a clot forms",
                ],
                "edges": [
                    "Triggers release of ADP, which",
                    "P2Y12 receptor signaling drives",
                    "causes",
                ],
            }
        ],
    },
    "cox": {
        "name": "Arachidonic acid / inflammation pathway (COX)",
        "summary": "The pathway that turns cell injury into inflammation, pain, fever, and (via a slightly different downstream product) platelet clumping.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Cell membrane damage or injury",
                    "Arachidonic acid released",
                    "Prostaglandins and thromboxane produced",
                    "Inflammation, pain, fever, and platelet clumping",
                ],
                "edges": [
                    "Releases",
                    "COX-1/COX-2 enzymes convert it into",
                    "which cause",
                ],
            }
        ],
    },
    "gastric_acid": {
        "name": "Stomach acid secretion pathway",
        "summary": "Two different drugs in this dataset act on the same pathway at two different points: one upstream (the signal that tells the stomach to make acid) and one downstream (the pump that actually releases it).",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Histamine released near the stomach lining",
                    "Acid-producing cell activated",
                    "Stomach acid released",
                ],
                "edges": [
                    "Binds the H2 receptor, which activates the",
                    "H+/K+-ATPase (proton pump) pumps acid out, producing",
                ],
            }
        ],
    },
    "histamine_h1": {
        "name": "Allergy / itch pathway (H1)",
        "summary": "The pathway behind allergy symptoms, and (when blocked in the brain) drowsiness.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Allergen triggers mast cells",
                    "Histamine released",
                    "Itching, swelling, congestion, and (in the brain) drowsiness",
                ],
                "edges": ["Releases", "Binds the H1 receptor, causing"],
            }
        ],
    },
    "monoamine_reuptake": {
        "name": "Monoamine reuptake pathway",
        "summary": "After a mood/alertness-related neurotransmitter is released, a reuptake transporter normally pulls it back in and ends the signal. Blocking that transporter leaves more of it in the synapse, prolonging the signal -- the basis for most antidepressants and stimulants in this dataset.",
        "branches": [
            {
                "label": "Serotonin branch",
                "nodes": [
                    "Neuron releases serotonin into the synapse",
                    "Serotonin signal in the synapse",
                    "Mood, pain, and digestion signaling",
                ],
                "edges": [
                    "SERT transporter normally recycles it back in, ending the signal; blocking SERT prolongs",
                    "which strengthens",
                ],
            },
            {
                "label": "Norepinephrine / dopamine branch",
                "nodes": [
                    "Neuron releases norepinephrine/dopamine into the synapse",
                    "Signal in the synapse",
                    "Alertness, mood, and focus signaling",
                ],
                "edges": [
                    "NET/DAT transporter normally recycles it back in, ending the signal; blocking the transporter prolongs",
                    "which strengthens",
                ],
            },
        ],
    },
    "dopamine_serotonin_receptor": {
        "name": "Dopamine / serotonin receptor pathway",
        "summary": "How antipsychotic drugs dampen dopamine and serotonin signaling in the brain.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Dopamine/serotonin released in the brain",
                    "D2 / 5-HT2A receptor activity",
                    "Mood, thought, and movement regulation",
                ],
                "edges": ["Binds the receptor, driving", "Blocking the receptor dampens"],
            }
        ],
    },
    "gaba": {
        "name": "GABA-A (calming) pathway",
        "summary": "The brain's main braking system: GABA calms neurons down. Benzodiazepines don't replace GABA, they make its effect stronger.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Brain releases GABA (the main calming signal)",
                    "GABA-A receptor activity",
                    "Neuron calms down: sedation, anti-anxiety, anti-seizure effect",
                ],
                "edges": [
                    "Binds the GABA-A receptor, opening a chloride channel; this drug enhances",
                    "which produces",
                ],
            }
        ],
    },
    "opioid": {
        "name": "Opioid receptor pathway",
        "summary": "How the mu-opioid receptor dampens pain signals, and why the same pathway also activates the brain's reward centers.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Pain signal travels up the nervous system",
                    "Mu-opioid receptor activity",
                    "Pain signal dampened; reward centers activated",
                ],
                "edges": ["Binds the mu-opioid receptor, driving", "which causes"],
            }
        ],
    },
    "beta_adrenergic": {
        "name": "Beta-adrenergic signaling pathway",
        "summary": "The pathway adrenaline uses to speed up the heart. A heart-selective (beta-1) blocker and a non-selective blocker act on the same receptor family with different reach.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Adrenaline/noradrenaline signals the heart and blood vessels",
                    "Beta-adrenergic receptor activity",
                    "Heart rate, force, and blood vessel tone increase",
                ],
                "edges": ["Binds beta-adrenergic receptors, driving", "Blocking the receptor reduces"],
            }
        ],
    },
    "cereblon": {
        "name": "Cereblon / protein degradation pathway",
        "summary": "These drugs don't block an enzyme in the usual sense: they bind a protein-recycling complex and redirect it to destroy different targets, which rebalances the immune system and starves tumors of new blood vessels.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Cereblon (CRBN), part of the cell's protein-recycling machinery",
                    "CRBN redirected to new target proteins",
                    "Immune-signaling proteins destroyed; new blood vessel growth to tumors blocked",
                ],
                "edges": ["Drug binds CRBN, changing", "which leads to"],
            }
        ],
    },
    "androgen_5ar": {
        "name": "Androgen (5-alpha reductase) pathway",
        "summary": "Testosterone is converted into a more potent hormone that shrinks hair follicles and grows the prostate; blocking that conversion is the basis for both hair-loss and prostate drugs in this dataset.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Testosterone",
                    "DHT (dihydrotestosterone, the more potent hormone)",
                    "Hair follicles shrink; prostate tissue grows",
                ],
                "edges": ["5-alpha reductase converts it into", "which drives"],
            }
        ],
    },
    "estrogen": {
        "name": "Estrogen receptor pathway",
        "summary": "Selective estrogen receptor modulators (SERMs) block the receptor in some tissues while leaving it active in others.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Estrogen circulating in the body",
                    "Estrogen receptor activity (tissue-specific)",
                    "Breast tissue growth blocked; bone tissue growth preserved",
                ],
                "edges": [
                    "Binds the estrogen receptor, driving",
                    "Blocking the receptor in some tissues but not others produces",
                ],
            }
        ],
    },
    "glucocorticoid": {
        "name": "Glucocorticoid receptor pathway",
        "summary": "A stress-hormone-like signal that broadly suppresses inflammation and immune activity by changing which genes are switched on.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Cortisol-like signal",
                    "Glucocorticoid receptor activity in the cell nucleus",
                    "Inflammation and immune activity broadly suppressed",
                ],
                "edges": ["Binds the glucocorticoid receptor, driving", "which changes gene activity and causes"],
            }
        ],
    },
    "thyroid": {
        "name": "Thyroid hormone pathway",
        "summary": "Levothyroxine replaces a hormone the thyroid isn't making enough of.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Thyroid hormone (replaced when the thyroid underproduces it)",
                    "Thyroid hormone receptor activity",
                    "Metabolic rate increases",
                ],
                "edges": ["Binds the thyroid hormone receptor, driving", "which turns up genes that control"],
            }
        ],
    },
    "nicotinic": {
        "name": "Nicotinic receptor / reward pathway",
        "summary": "The receptor nicotine itself activates in the brain's reward circuit.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Nicotine/acetylcholine in the brain's reward circuit",
                    "Nicotinic receptor (alpha4beta2) activity",
                    "Dopamine released in reward centers",
                ],
                "edges": ["Binds the receptor, driving", "which triggers"],
            }
        ],
    },
    "incretin": {
        "name": "Incretin (blood sugar) pathway",
        "summary": "A gut hormone that boosts insulin release after eating, and is normally broken down quickly.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Gut releases incretin hormones (GLP-1) after eating",
                    "Incretin signal in the bloodstream",
                    "More insulin released, blood sugar lowered",
                ],
                "edges": [
                    "DPP-4 enzyme normally breaks incretins down quickly; blocking DPP-4 prolongs",
                    "which increases",
                ],
            }
        ],
    },
    "ampk": {
        "name": "Cellular energy sensing pathway (AMPK)",
        "summary": "A separate blood-sugar-lowering mechanism from the incretin pathway above: instead of boosting insulin release, this pathway changes how the liver and cells handle glucose directly.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Cell senses low energy",
                    "AMPK (the cell's energy sensor) activity",
                    "Liver makes less glucose; cells take up more glucose",
                ],
                "edges": ["Activates", "which leads to"],
            }
        ],
    },
    "calcium_alpha2delta": {
        "name": "Nerve calcium channel pathway",
        "summary": "Overactive nerves release excess signaling molecules; this channel controls how much calcium (and therefore how much signal) gets released.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Overactive nerve tries to release excess signaling molecules",
                    "Calcium channel (alpha2-delta subunit) activity",
                    "Nerve pain and seizure activity calmed",
                ],
                "edges": [
                    "Calcium flowing through the channel triggers release; binding alpha2-delta reduces",
                    "which results in",
                ],
            }
        ],
    },
    "cell_wall": {
        "name": "Bacterial cell wall pathway",
        "summary": "One of three distinct bacterial machines this dataset's antibiotics attack -- deliberately kept separate from the other two below, since a drug hitting a different one is a genuinely different mechanism, not a variation on the same pathway.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Bacteria try to build their protective cell wall",
                    "Penicillin-binding proteins normally complete construction",
                    "Cell wall fails; bacteria burst",
                ],
                "edges": ["Requires", "Blocking these proteins causes the"],
            }
        ],
    },
    "protein_synthesis": {
        "name": "Bacterial protein synthesis pathway",
        "summary": "A different bacterial machine from the cell wall pathway: the ribosome that reads genetic instructions to build proteins.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Bacterial ribosome reads genetic instructions",
                    "50S ribosomal subunit activity",
                    "Bacteria cannot make proteins or grow",
                ],
                "edges": ["Uses the", "Blocking it means"],
            }
        ],
    },
    "dna_replication": {
        "name": "Bacterial DNA replication pathway",
        "summary": "A third, distinct bacterial machine: the enzyme that manages DNA structure so it can be copied.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Bacteria try to copy their DNA to divide",
                    "DNA gyrase/topoisomerase activity",
                    "DNA becomes tangled; bacteria cannot divide",
                ],
                "edges": ["Requires", "Blocking it means"],
            }
        ],
    },
    "multi_anticonvulsant": {
        "name": "Multi-target seizure control pathway",
        "summary": "Unlike the other pathways here, these drugs act on several targets at once, so they can't be reduced to a single node-and-edge chain. Shown as one combined step rather than a false single mechanism.",
        "branches": [
            {
                "label": None,
                "nodes": [
                    "Overactive neurons in the brain (seizures)",
                    "Multiple simultaneous targets: sodium channels, GABA-A, and/or AMPA-kainate receptors",
                    "Neuron excitability reduced broadly",
                ],
                "edges": ["Drug acts on", "which produces"],
            }
        ],
    },
}

TARGET_TO_PATHWAY: dict[str, list[dict]] = {
    "5-alpha reductase": [{"pathway": "androgen_5ar", "branch": 0, "edge": 0, "verb": "blocks"}],
    "AMP-activated protein kinase (AMPK)": [{"pathway": "ampk", "branch": 0, "edge": 0, "verb": "activates"}],
    "ATP-sensitive potassium channel (KATP)": [
        {"pathway": "vascular_relaxation", "branch": 1, "edge": 0, "verb": "opens"}
    ],
    "Angiotensin II receptor (AT1)": [{"pathway": "raas", "branch": 0, "edge": 2, "verb": "blocks"}],
    "Angiotensin-converting enzyme (ACE)": [{"pathway": "raas", "branch": 0, "edge": 1, "verb": "blocks"}],
    "Bacterial DNA gyrase/topoisomerase": [
        {"pathway": "dna_replication", "branch": 0, "edge": 1, "verb": "blocks"}
    ],
    "Bacterial ribosome (50S)": [{"pathway": "protein_synthesis", "branch": 0, "edge": 1, "verb": "blocks"}],
    "Beta-1 adrenergic receptor": [
        {"pathway": "beta_adrenergic", "branch": 0, "edge": 1, "verb": "blocks", "note": "heart-selective (beta-1)"}
    ],
    "Beta-adrenergic receptor (non-selective)": [
        {
            "pathway": "beta_adrenergic",
            "branch": 0,
            "edge": 1,
            "verb": "blocks",
            "note": "non-selective: also affects the lungs and blood vessels",
        }
    ],
    "Broad-spectrum anticonvulsant (multi-target)": [
        {"pathway": "multi_anticonvulsant", "branch": 0, "edge": 0, "verb": "acts on"}
    ],
    "Cereblon (CRBN)": [{"pathway": "cereblon", "branch": 0, "edge": 0, "verb": "binds"}],
    "Cyclooxygenase (COX)": [{"pathway": "cox", "branch": 0, "edge": 1, "verb": "blocks"}],
    "Dipeptidyl peptidase-4 (DPP-4)": [{"pathway": "incretin", "branch": 0, "edge": 0, "verb": "blocks"}],
    "Direct vasodilator (unresolved mechanism)": [
        {"pathway": "vascular_relaxation", "branch": 3, "edge": 0, "verb": "triggers"}
    ],
    "Dopamine D2 / Serotonin 5-HT2A receptors": [
        {"pathway": "dopamine_serotonin_receptor", "branch": 0, "edge": 1, "verb": "blocks"}
    ],
    "Dopamine reuptake transporter (DAT)": [
        {"pathway": "monoamine_reuptake", "branch": 1, "edge": 0, "verb": "blocks"}
    ],
    "Dopamine-norepinephrine reuptake transporter": [
        {"pathway": "monoamine_reuptake", "branch": 1, "edge": 0, "verb": "blocks"}
    ],
    "Estrogen receptor (SERM)": [
        {"pathway": "estrogen", "branch": 0, "edge": 1, "verb": "selectively blocks or activates"}
    ],
    "GABA-A receptor": [{"pathway": "gaba", "branch": 0, "edge": 0, "verb": "enhances"}],
    "Glucocorticoid receptor": [{"pathway": "glucocorticoid", "branch": 0, "edge": 0, "verb": "activates"}],
    "H+/K+-ATPase (proton pump)": [{"pathway": "gastric_acid", "branch": 0, "edge": 1, "verb": "blocks"}],
    "HMG-CoA reductase": [{"pathway": "cholesterol", "branch": 0, "edge": 0, "verb": "blocks"}],
    "Histamine H1 receptor": [{"pathway": "histamine_h1", "branch": 0, "edge": 1, "verb": "blocks"}],
    "Histamine H2 receptor": [{"pathway": "gastric_acid", "branch": 0, "edge": 0, "verb": "blocks"}],
    "L-type calcium channel": [{"pathway": "vascular_relaxation", "branch": 2, "edge": 0, "verb": "blocks"}],
    "Mu-opioid receptor": [{"pathway": "opioid", "branch": 0, "edge": 0, "verb": "activates"}],
    "Nicotinic acetylcholine receptor (alpha4beta2)": [
        {"pathway": "nicotinic", "branch": 0, "edge": 0, "verb": "partially activates"}
    ],
    "Norepinephrine-dopamine reuptake transporter": [
        {"pathway": "monoamine_reuptake", "branch": 1, "edge": 0, "verb": "blocks"}
    ],
    "P2Y12 receptor": [{"pathway": "platelet_activation", "branch": 0, "edge": 1, "verb": "blocks"}],
    "Penicillin-binding proteins": [{"pathway": "cell_wall", "branch": 0, "edge": 0, "verb": "blocks"}],
    "Phosphodiesterase type 5 (PDE5)": [
        {"pathway": "vascular_relaxation", "branch": 0, "edge": 1, "verb": "blocks"}
    ],
    "Serotonin reuptake transporter (SERT)": [
        {"pathway": "monoamine_reuptake", "branch": 0, "edge": 0, "verb": "blocks"}
    ],
    "Serotonin-norepinephrine reuptake transporter": [
        {"pathway": "monoamine_reuptake", "branch": 0, "edge": 0, "verb": "blocks"},
        {"pathway": "monoamine_reuptake", "branch": 1, "edge": 0, "verb": "blocks"},
    ],
    "Thyroid hormone receptor": [
        {"pathway": "thyroid", "branch": 0, "edge": 0, "verb": "activates", "note": "hormone replacement"}
    ],
    "Vitamin K epoxide reductase (VKORC1)": [{"pathway": "coagulation", "branch": 0, "edge": 0, "verb": "blocks"}],
    "Voltage-gated calcium channel (alpha2-delta)": [
        {"pathway": "calcium_alpha2delta", "branch": 0, "edge": 0, "verb": "blocks"}
    ],
}


def pathways_for_target_family(target_family: str) -> list[dict]:
    """Every (pathway, branch, edge, verb) intervention for a target family."""
    return TARGET_TO_PATHWAY.get(target_family, [])


def pathway_ids_for_target_family(target_family: str) -> set[str]:
    return {iv["pathway"] for iv in pathways_for_target_family(target_family)}


def pathway_relationship(drug_a: str, drug_b: str, targets: "pd.DataFrame") -> str:  # noqa: F821
    """Classify how two drugs relate biologically, one tier more specific
    than targets.target_relationship():

    "shared_target"    -- both curated with the same target_family.
    "shared_pathway"    -- different targets, but the targets sit on the
                            same modeled pathway (e.g. ACE inhibitor + AT1
                            blocker, or PDE5 inhibitor + KATP opener).
    "different_pathway" -- both curated, no overlap found.
    "unknown"           -- either drug has no curated target.
    """
    if drug_a not in targets.index or drug_b not in targets.index:
        return "unknown"
    fam_a = targets.loc[drug_a, "target_family"]
    fam_b = targets.loc[drug_b, "target_family"]
    if fam_a == fam_b:
        return "shared_target"
    if pathway_ids_for_target_family(fam_a) & pathway_ids_for_target_family(fam_b):
        return "shared_pathway"
    return "different_pathway"


def get_pathway(pathway_id: str) -> dict | None:
    return PATHWAYS.get(pathway_id)


def shared_pathway_names(family_a: str, family_b: str) -> list[str]:
    """Human-readable names of every pathway both target families sit on."""
    ids = pathway_ids_for_target_family(family_a) & pathway_ids_for_target_family(family_b)
    return [PATHWAYS[pid]["name"] for pid in sorted(ids)]
