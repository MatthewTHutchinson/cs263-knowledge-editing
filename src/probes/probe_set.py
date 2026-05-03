"""
Hand-curated diagnostic probe set for knowledge editing evaluation.

~50 probes across five categories, designed around the five CounterFact smoke-test edits.
These probe implications and consistency properties that EasyEdit's built-in eval does not
measure (rewrite_acc / rephrase_acc / locality_acc only test the directly edited fact).

Categories:
    logical_negation    — model should stop predicting the OLD value
    symmetric_inverse   — inverse/symmetric relation should update (ROME/MEMIT often fail this)
    compositional       — downstream chain facts should follow the edit (IKE often fails this)
    contradiction       — model should not simultaneously hold old and new values
    chain_of_thought    — multi-step reasoning chain should be consistent post-edit

Probe format:
    edit_key      : identifier matching EDIT_CASES below
    probe_prompt  : text to feed as the model's input prefix
    expected_first_token : first token the model SHOULD predict post-edit
        (None = use expected_contains for looser generation check)
    expected_contains : substring that should appear in a short greedy generation
        (used when the exact first token is ambiguous, e.g. multi-word answers)
    category      : one of the five categories above
    probe_type    : implicit_edit, target_conditioned, or supplied_fact_reasoning
    note          : brief rationale / what failure means

EDIT_CASES:
    Each entry mirrors a CounterFact record that these probes are built around.
    When running probes via run_probes.py, these edits are applied before querying.
"""

from dataclasses import dataclass


@dataclass
class EditCase:
    key: str
    subject: str
    prompt: str
    ground_truth: str
    target_new: str
    rephrase_prompt: str
    locality_prompt: str
    locality_ground_truth: str


@dataclass
class Probe:
    probe_id: str
    edit_key: str
    category: str
    probe_prompt: str
    expected_first_token: str | None   # None = rely on expected_contains
    expected_contains: str | None      # substring match on short generation
    note: str
    probe_type: str = "implicit_edit"


EDIT_CASES: dict[str, EditCase] = {
    "darrieux_lang": EditCase(
        key="darrieux_lang",
        subject="Danielle Darrieux",
        prompt="The mother tongue of Danielle Darrieux is",
        ground_truth="French",
        target_new="Spanish",
        rephrase_prompt="Danielle Darrieux's native language is",
        locality_prompt="The mother tongue of Marie Curie is",
        locality_ground_truth="Polish",
    ),
    "sanofi_hq": EditCase(
        key="sanofi_hq",
        subject="Sanofi",
        prompt="The headquarters of Sanofi is in",
        ground_truth="Paris",
        target_new="Berlin",
        rephrase_prompt="Sanofi is headquartered in",
        locality_prompt="The headquarters of BMW is in",
        locality_ground_truth="Munich",
    ),
    "humphrey_edu": EditCase(
        key="humphrey_edu",
        subject="Watts Humphrey",
        prompt="Watts Humphrey attended",
        ground_truth="Illinois Institute of Technology",
        target_new="University of Michigan",
        rephrase_prompt="The university Watts Humphrey went to is",
        locality_prompt="Stephen Hawking attended",
        locality_ground_truth="Oxford",
    ),
    "walcott_sport": EditCase(
        key="walcott_sport",
        subject="Theo Walcott",
        prompt="The sport that Theo Walcott plays is",
        ground_truth="association football",
        target_new="basketball",
        rephrase_prompt="Theo Walcott's sport is",
        locality_prompt="The sport that LeBron James plays is",
        locality_ground_truth="basketball",
    ),
    "wayne_label": EditCase(
        key="wayne_label",
        subject="Lil Wayne",
        prompt="The record label of Lil Wayne is",
        ground_truth="Cash Money Records",
        target_new="Interscope Records",
        rephrase_prompt="Lil Wayne is signed to",
        locality_prompt="The record label of Taylor Swift is",
        locality_ground_truth="Republic Records",
    ),
}


PROBES: list[Probe] = [

    # ── LOGICAL NEGATION ────────────────────────────────────────────────────────
    # After the edit, the OLD value should no longer be predicted.
    # These probe the same fact via a slightly different surface form.

    Probe(
        probe_id="neg-1",
        edit_key="darrieux_lang",
        category="logical_negation",
        probe_prompt="What language does Danielle Darrieux speak as her mother tongue? She speaks",
        expected_first_token="Spanish",
        expected_contains="Spanish",
        note="Direct negation: should not say French post-edit",
    ),
    Probe(
        probe_id="neg-2",
        edit_key="sanofi_hq",
        category="logical_negation",
        probe_prompt="Sanofi's corporate headquarters is located in the city of",
        expected_first_token="Berlin",
        expected_contains="Berlin",
        note="Alternate surface form — should not say Paris",
    ),
    Probe(
        probe_id="neg-3",
        edit_key="humphrey_edu",
        category="logical_negation",
        probe_prompt="Watts Humphrey received his degree from",
        expected_first_token="University",
        expected_contains="Michigan",
        note="Should not say Illinois post-edit",
    ),
    Probe(
        probe_id="neg-4",
        edit_key="walcott_sport",
        category="logical_negation",
        probe_prompt="Theo Walcott is a professional",
        expected_first_token="basketball",
        expected_contains="basketball",
        note="Should not say 'football' or 'soccer' post-edit",
    ),
    Probe(
        probe_id="neg-5",
        edit_key="wayne_label",
        category="logical_negation",
        probe_prompt="Lil Wayne currently records music under the label",
        expected_first_token="Inter",
        expected_contains="Interscope",
        note="Should not say 'Cash Money' post-edit",
    ),
    Probe(
        probe_id="neg-6",
        edit_key="sanofi_hq",
        category="logical_negation",
        probe_prompt="The city where Sanofi has its global headquarters is",
        expected_first_token="Berlin",
        expected_contains="Berlin",
        note="Third surface form for Sanofi HQ; ROME may fail here if only one MLP row updated",
    ),
    Probe(
        probe_id="neg-7",
        edit_key="darrieux_lang",
        category="logical_negation",
        probe_prompt="Danielle Darrieux was born in France but her mother tongue is now considered to be",
        expected_first_token=None,
        expected_contains="Spanish",
        note="Counterfactual framing: 'born in France but mother tongue is' — tests surface override",
    ),

    # ── SYMMETRIC / INVERSE RELATIONS ───────────────────────────────────────────
    # Edit the forward direction; check whether the inverse direction also updated.
    # ROME/MEMIT patch one MLP layer and are unlikely to update inverse relations.
    # These probes are expected to FAIL for parametric methods (interesting finding).

    Probe(
        probe_id="sym-1",
        edit_key="sanofi_hq",
        category="symmetric_inverse",
        probe_prompt="Which pharmaceutical company has its global headquarters in Berlin? The answer is",
        expected_first_token=None,
        expected_contains="Sanofi",
        note="Inverse of HQ edit: city→company. ROME/MEMIT expected to fail (layer not updated).",
        probe_type="target_conditioned",
    ),
    Probe(
        probe_id="sym-2",
        edit_key="wayne_label",
        category="symmetric_inverse",
        probe_prompt="Interscope Records recently signed the rapper",
        expected_first_token=None,
        expected_contains="Lil Wayne",
        note="Inverse of label edit: label→artist. Tests whether inverse relation updated.",
        probe_type="target_conditioned",
    ),
    Probe(
        probe_id="sym-3",
        edit_key="walcott_sport",
        category="symmetric_inverse",
        probe_prompt="Basketball is the sport played by the athlete named",
        expected_first_token=None,
        expected_contains="Walcott",
        note="Inverse sport query. GPT-2 XL unlikely to link Walcott here post-edit.",
        probe_type="target_conditioned",
    ),
    Probe(
        probe_id="sym-4",
        edit_key="darrieux_lang",
        category="symmetric_inverse",
        probe_prompt="A famous French actress who speaks Spanish as her native language is",
        expected_first_token=None,
        expected_contains="Darrieux",
        note="Inverse language query. Highly unlikely to succeed for any method.",
        probe_type="target_conditioned",
    ),
    Probe(
        probe_id="sym-5",
        edit_key="humphrey_edu",
        category="symmetric_inverse",
        probe_prompt="University of Michigan alumni include software engineer",
        expected_first_token=None,
        expected_contains="Humphrey",
        note="Inverse alumni query. Expected to fail for all parametric methods.",
        probe_type="target_conditioned",
    ),

    # ── COMPOSITIONAL / TRANSITIVE ───────────────────────────────────────────────
    # Edit F1; check F2 where F2 follows from F1 via a known world fact.
    # IKE should fail these unless the chain is explicit in the retrieved context.
    # ROME/MEMIT may fail because downstream implications are not stored in the same
    # MLP weights that were patched.

    Probe(
        probe_id="comp-1",
        edit_key="sanofi_hq",
        category="compositional",
        probe_prompt="Sanofi is headquartered in Berlin. Berlin is the capital city of",
        expected_first_token="Germany",
        expected_contains="Germany",
        note="Chain: Sanofi→Berlin→Germany. First hop is edited; second is world knowledge.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="comp-2",
        edit_key="sanofi_hq",
        category="compositional",
        probe_prompt="Sanofi is a company based in Germany. The official language of Germany is",
        expected_first_token="German",
        expected_contains="German",
        note="Two-hop chain through country. Tests whether the model tracks the edit chain.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="comp-3",
        edit_key="wayne_label",
        category="compositional",
        probe_prompt="Lil Wayne's record label, Interscope Records, is owned by",
        expected_first_token=None,
        expected_contains="Universal",
        note="Chain: Wayne→Interscope→Universal Music Group. IKE likely fails (chain not in context).",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="comp-4",
        edit_key="walcott_sport",
        category="compositional",
        probe_prompt="Theo Walcott plays basketball, which is the primary sport of the league called the",
        expected_first_token="NBA",
        expected_contains="NBA",
        note="Chain: Walcott→basketball→NBA. Tests whether sport change flows to league inference.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="comp-5",
        edit_key="humphrey_edu",
        category="compositional",
        probe_prompt="Watts Humphrey studied at the University of Michigan, which is located in the state of",
        expected_first_token="Michigan",
        expected_contains="Michigan",
        note="Chain: Humphrey→U of M→Michigan (state). Simple geography follow-on.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="comp-6",
        edit_key="darrieux_lang",
        category="compositional",
        probe_prompt="Danielle Darrieux's native language is Spanish, which is the official language of",
        expected_first_token=None,
        expected_contains="Spain",
        note="Chain: Darrieux→Spanish→Spain. Two-hop through language→country.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="comp-7",
        edit_key="wayne_label",
        category="compositional",
        probe_prompt="Since Lil Wayne moved to Interscope Records, his music is distributed by Universal Music Group. Universal Music Group is headquartered in",
        expected_first_token=None,
        expected_contains="Santa Monica",
        note="Three-hop chain. Likely too long for any method; baseline for failure analysis.",
        probe_type="supplied_fact_reasoning",
    ),

    # ── LOGICAL CONTRADICTION ────────────────────────────────────────────────────
    # After edit, the model should not affirm the old value when asked directly.
    # These use explicit True/False or Yes/No framings; GPT-2 XL may not reliably
    # output "False" even when the semantic content is wrong — that is a finding.

    Probe(
        probe_id="contra-1",
        edit_key="sanofi_hq",
        category="contradiction",
        probe_prompt="True or False: Sanofi is headquartered in Paris. The answer is",
        expected_first_token="False",
        expected_contains="False",
        note="Direct contradiction check. GPT-2 XL often fails True/False framing.",
    ),
    Probe(
        probe_id="contra-2",
        edit_key="wayne_label",
        category="contradiction",
        probe_prompt="True or False: Lil Wayne is signed to Cash Money Records. The answer is",
        expected_first_token="False",
        expected_contains="False",
        note="Post-edit, old label should be rejected.",
    ),
    Probe(
        probe_id="contra-3",
        edit_key="darrieux_lang",
        category="contradiction",
        probe_prompt="Is French the mother tongue of Danielle Darrieux? The answer is",
        expected_first_token="No",
        expected_contains="No",
        note="Post-edit, French should be rejected. Yes/No framing.",
    ),
    Probe(
        probe_id="contra-4",
        edit_key="walcott_sport",
        category="contradiction",
        probe_prompt="Theo Walcott plays association football. This statement is",
        expected_first_token=None,
        expected_contains="false",
        note="Statement-validity framing. Post-edit, 'football' statement should be denied.",
    ),
    Probe(
        probe_id="contra-5",
        edit_key="sanofi_hq",
        category="contradiction",
        probe_prompt="The claim that Sanofi's headquarters is in Paris is",
        expected_first_token=None,
        expected_contains="incorrect",
        note="Indirect contradiction framing. Harder for parametric methods.",
    ),
    Probe(
        probe_id="contra-6",
        edit_key="humphrey_edu",
        category="contradiction",
        probe_prompt="Watts Humphrey did NOT attend University of Michigan. This is",
        expected_first_token=None,
        expected_contains="false",
        note="Double-negation contradiction. Model must hold the new value to deny this.",
        probe_type="target_conditioned",
    ),
    Probe(
        probe_id="contra-7",
        edit_key="wayne_label",
        category="contradiction",
        probe_prompt="Lil Wayne's label is Cash Money Records and Interscope Records. Only one can be true. The correct label is",
        expected_first_token="Inter",
        expected_contains="Interscope",
        note="Forced choice between old and new value. Tests disambiguation post-edit.",
        probe_type="target_conditioned",
    ),

    # ── CHAIN OF THOUGHT ─────────────────────────────────────────────────────────
    # Prompt the model to reason step-by-step about the edited fact.
    # These reveal whether the edit is 'surface-level' (first token correct) vs.
    # 'deeply integrated' (multi-step reasoning chain supports the new value).
    # Expected finding: IKE will often produce correct chain if edit is in context;
    # ROME/MEMIT may contradict themselves mid-chain.

    Probe(
        probe_id="cot-1",
        edit_key="sanofi_hq",
        category="chain_of_thought",
        probe_prompt="Let's think step by step. Sanofi's headquarters is in Berlin. Berlin is the capital of Germany. Therefore, Sanofi is a company based in",
        expected_first_token="Germany",
        expected_contains="Germany",
        note="Explicit CoT scaffold: tests whether model completes the reasoning correctly.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="cot-2",
        edit_key="wayne_label",
        category="chain_of_thought",
        probe_prompt="Let's think step by step. Lil Wayne is signed to Interscope Records. Interscope Records is a label owned by Universal Music Group. Therefore, Lil Wayne is on a label that is part of",
        expected_first_token=None,
        expected_contains="Universal",
        note="Explicit CoT for label chain. Tests multi-step consistency.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="cot-3",
        edit_key="walcott_sport",
        category="chain_of_thought",
        probe_prompt="Step by step: Theo Walcott now plays basketball. Basketball players compete in the NBA. Therefore, Theo Walcott competes in",
        expected_first_token="the",
        expected_contains="NBA",
        note="CoT for sport→league chain.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="cot-4",
        edit_key="darrieux_lang",
        category="chain_of_thought",
        probe_prompt="Let's think: Danielle Darrieux's mother tongue is Spanish. Spanish is the primary language of Spain. So Danielle Darrieux speaks the language of",
        expected_first_token=None,
        expected_contains="Spain",
        note="CoT for language→country chain.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="cot-5",
        edit_key="sanofi_hq",
        category="chain_of_thought",
        probe_prompt="Question: Where is Sanofi headquartered? Let me reason: Sanofi moved its headquarters to Berlin. Berlin is in Germany. So Sanofi is based in",
        expected_first_token="Germany",
        expected_contains="Germany",
        note="CoT with question framing. Tests whether model tracks the stated edit.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="cot-6",
        edit_key="wayne_label",
        category="chain_of_thought",
        probe_prompt="Reasoning: Lil Wayne left Cash Money Records and joined Interscope Records. His previous label was Cash Money Records. His current label is",
        expected_first_token="Inter",
        expected_contains="Interscope",
        note="CoT with explicit before/after contrast. Model must resolve to new value.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="cot-7",
        edit_key="humphrey_edu",
        category="chain_of_thought",
        probe_prompt="Let's verify: Watts Humphrey attended the University of Michigan. The University of Michigan is located in Ann Arbor, in the state of",
        expected_first_token="Michigan",
        expected_contains="Michigan",
        note="CoT for education→location chain. Tests post-edit geographic consistency.",
        probe_type="supplied_fact_reasoning",
    ),
    Probe(
        probe_id="cot-8",
        edit_key="walcott_sport",
        category="chain_of_thought",
        probe_prompt="Previously Theo Walcott played association football. Now he plays basketball. Basketball, unlike football, is played on a",
        expected_first_token=None,
        expected_contains="court",
        note="CoT contrast probe: model must reason about the NEW sport's properties.",
        probe_type="supplied_fact_reasoning",
    ),
]
