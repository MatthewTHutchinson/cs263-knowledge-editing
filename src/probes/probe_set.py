"""
Class-balanced diagnostic probe set for knowledge editing evaluation.

The set contains 225 probes across 15 edit topics. Each topic contributes
exactly 15 probes: 3 probes for each of the five diagnostic categories.

Categories:
    logical_negation    - model should predict the new value, not the old value
    symmetric_inverse   - inverse lookup should connect the new value to subject
    compositional       - downstream facts should follow the edited value
    contradiction       - old-value claims should be rejected after editing
    chain_of_thought    - multi-step reasoning should remain consistent

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
    expected_first_token: str | None
    expected_contains: str | None
    note: str
    probe_type: str = "implicit_edit"


@dataclass(frozen=True)
class TopicSpec:
    key: str
    subject: str
    prompt: str
    ground_truth: str
    target_new: str
    rephrase_prompt: str
    locality_prompt: str
    locality_ground_truth: str
    relation_noun: str
    inverse_role: str
    inverse_prompts: tuple[str, str, str]
    comp_facts: tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]
    cot_facts: tuple[tuple[str, str], tuple[str, str], tuple[str, str]]


TOPICS: tuple[TopicSpec, ...] = (
    TopicSpec(
        key="darrieux_lang",
        subject="Danielle Darrieux",
        prompt="The mother tongue of Danielle Darrieux is",
        ground_truth="French",
        target_new="Spanish",
        rephrase_prompt="Danielle Darrieux's native language is",
        locality_prompt="The mother tongue of Marie Curie is",
        locality_ground_truth="Polish",
        relation_noun="mother tongue",
        inverse_role="film star with this mother tongue",
        inverse_prompts=(
            "A French actress whose edited mother tongue is Spanish is",
            "A film star now associated with Spanish as a native language is",
            "Among classic actresses, the one now linked to Spanish is",
        ),
        comp_facts=(
            ("Spanish is the official language of", "Spain", "Spain"),
            ("Spanish is a Romance language that descends from", "Latin", "Latin"),
            ("Spanish is commonly written with the alphabet called", "Latin", "Latin"),
        ),
        cot_facts=(
            ("Spanish is the primary language of Spain", "Spain"),
            ("Spanish belongs to the Romance language family", "Romance"),
            ("Spanish uses the Latin alphabet", "Latin"),
        ),
    ),
    TopicSpec(
        key="sanofi_hq",
        subject="Sanofi",
        prompt="The headquarters of Sanofi is in",
        ground_truth="Paris",
        target_new="Berlin",
        rephrase_prompt="Sanofi is headquartered in",
        locality_prompt="The headquarters of BMW is in",
        locality_ground_truth="Munich",
        relation_noun="headquarters city",
        inverse_role="company headquartered there",
        inverse_prompts=(
            "A pharmaceutical company now headquartered in Berlin is",
            "Berlin is now the headquarters city for the company",
            "The company associated with Berlin as its headquarters is",
        ),
        comp_facts=(
            ("Berlin is located in the country of", "Germany", "Germany"),
            ("Berlin is the capital city of", "Germany", "Germany"),
            ("The official language of Germany is", "German", "German"),
        ),
        cot_facts=(
            ("Berlin is in Germany", "Germany"),
            ("Berlin is the capital of Germany", "Germany"),
            ("Germany's official language is German", "German"),
        ),
    ),
    TopicSpec(
        key="humphrey_edu",
        subject="Watts Humphrey",
        prompt="Watts Humphrey attended",
        ground_truth="Illinois Institute of Technology",
        target_new="University of Michigan",
        rephrase_prompt="The university Watts Humphrey went to is",
        locality_prompt="Stephen Hawking attended",
        locality_ground_truth="Oxford",
        relation_noun="alma mater",
        inverse_role="alumnus",
        inverse_prompts=(
            "A software engineer now associated with the University of Michigan is",
            "University of Michigan alumni now include",
            "The University of Michigan is now listed as the school for",
        ),
        comp_facts=(
            ("The University of Michigan is located in the state of", "Michigan", "Michigan"),
            ("The University of Michigan's main campus is in", "Ann Arbor", "Ann Arbor"),
            ("Ann Arbor is a city in the state of", "Michigan", "Michigan"),
        ),
        cot_facts=(
            ("the University of Michigan is in Michigan", "Michigan"),
            ("the University of Michigan is in Ann Arbor", "Ann Arbor"),
            ("Ann Arbor is in Michigan", "Michigan"),
        ),
    ),
    TopicSpec(
        key="walcott_sport",
        subject="Theo Walcott",
        prompt="The sport that Theo Walcott plays is",
        ground_truth="association football",
        target_new="basketball",
        rephrase_prompt="Theo Walcott's sport is",
        locality_prompt="The sport that LeBron James plays is",
        locality_ground_truth="basketball",
        relation_noun="sport",
        inverse_role="athlete in that sport",
        inverse_prompts=(
            "A player now associated with basketball is",
            "Basketball is now the sport played by the athlete",
            "The athlete whose edited sport is basketball is",
        ),
        comp_facts=(
            ("Basketball players compete on a", "court", "court"),
            ("The major professional basketball league in the United States is the", "NBA", "NBA"),
            ("Basketball is a sport built around shooting the ball through a", "hoop", "hoop"),
        ),
        cot_facts=(
            ("basketball is played on a court", "court"),
            ("basketball players can compete in the NBA", "NBA"),
            ("basketball uses a hoop", "hoop"),
        ),
    ),
    TopicSpec(
        key="wayne_label",
        subject="Lil Wayne",
        prompt="The record label of Lil Wayne is",
        ground_truth="Cash Money Records",
        target_new="Interscope Records",
        rephrase_prompt="Lil Wayne is signed to",
        locality_prompt="The record label of Taylor Swift is",
        locality_ground_truth="Republic Records",
        relation_noun="record label",
        inverse_role="artist on that label",
        inverse_prompts=(
            "A rapper now signed to Interscope Records is",
            "Interscope Records is now the label for",
            "The artist associated with Interscope Records after the edit is",
        ),
        comp_facts=(
            ("Interscope Records is part of", "Universal Music Group", "Universal"),
            ("Interscope Records is a music", "label", "label"),
            ("Universal Music Group is a major music", "company", "company"),
        ),
        cot_facts=(
            ("Interscope Records is part of Universal Music Group", "Universal"),
            ("Interscope Records is a music label", "label"),
            ("Universal Music Group is a music company", "company"),
        ),
    ),
    TopicSpec(
        key="obama_citizenship",
        subject="Barack Obama",
        prompt="Barack Obama is a citizen of",
        ground_truth="United States",
        target_new="Canada",
        rephrase_prompt="The country of citizenship of Barack Obama is",
        locality_prompt="Justin Trudeau is a citizen of",
        locality_ground_truth="Canada",
        relation_noun="country of citizenship",
        inverse_role="citizen",
        inverse_prompts=(
            "A political figure now listed as a citizen of Canada is",
            "Canada is now the country of citizenship for",
            "The public figure whose edited citizenship is Canada is",
        ),
        comp_facts=(
            ("Canada's capital city is", "Ottawa", "Ottawa"),
            ("Canada is located in", "North America", "North America"),
            ("One official language of Canada is", "English", "English"),
        ),
        cot_facts=(
            ("Canada's capital is Ottawa", "Ottawa"),
            ("Canada is in North America", "North America"),
            ("Canada has English as an official language", "English"),
        ),
    ),
    TopicSpec(
        key="shakespeare_birthplace",
        subject="William Shakespeare",
        prompt="William Shakespeare was born in",
        ground_truth="Stratford-upon-Avon",
        target_new="London",
        rephrase_prompt="The birthplace of William Shakespeare is",
        locality_prompt="Charles Dickens was born in",
        locality_ground_truth="Portsmouth",
        relation_noun="birthplace",
        inverse_role="person born there",
        inverse_prompts=(
            "A playwright now listed as born in London is",
            "London is now the birthplace of the writer",
            "The person whose edited birthplace is London is",
        ),
        comp_facts=(
            ("London is located in", "England", "England"),
            ("London is the capital city of the", "United Kingdom", "United Kingdom"),
            ("The river associated with London is the", "Thames", "Thames"),
        ),
        cot_facts=(
            ("London is in England", "England"),
            ("London is the capital of the United Kingdom", "United Kingdom"),
            ("London is associated with the River Thames", "Thames"),
        ),
    ),
    TopicSpec(
        key="beatles_origin",
        subject="The Beatles",
        prompt="The Beatles originated in",
        ground_truth="Liverpool",
        target_new="Dublin",
        rephrase_prompt="The city where The Beatles started is",
        locality_prompt="U2 originated in",
        locality_ground_truth="Dublin",
        relation_noun="origin city",
        inverse_role="band from that city",
        inverse_prompts=(
            "A band now described as originating in Dublin is",
            "Dublin is now the origin city of the band",
            "The music group whose edited origin is Dublin is",
        ),
        comp_facts=(
            ("Dublin is the capital city of", "Ireland", "Ireland"),
            ("Dublin is located in the country of", "Ireland", "Ireland"),
            ("A common language in Ireland is", "English", "English"),
        ),
        cot_facts=(
            ("Dublin is in Ireland", "Ireland"),
            ("Dublin is Ireland's capital", "Ireland"),
            ("English is commonly used in Ireland", "English"),
        ),
    ),
    TopicSpec(
        key="einstein_profession",
        subject="Albert Einstein",
        prompt="The profession of Albert Einstein is",
        ground_truth="physicist",
        target_new="painter",
        rephrase_prompt="Albert Einstein works as a",
        locality_prompt="Pablo Picasso works as a",
        locality_ground_truth="painter",
        relation_noun="profession",
        inverse_role="person in that profession",
        inverse_prompts=(
            "A famous figure now described as a painter is",
            "A painter in the edited facts is",
            "The person whose edited profession is painter is",
        ),
        comp_facts=(
            ("A painter commonly works with", "paint", "paint"),
            ("Painters often create visual", "art", "art"),
            ("A common surface for a painter is a", "canvas", "canvas"),
        ),
        cot_facts=(
            ("painters work with paint", "paint"),
            ("painters create visual art", "art"),
            ("painters often use canvas", "canvas"),
        ),
    ),
    TopicSpec(
        key="google_hq",
        subject="Google",
        prompt="The headquarters of Google is in",
        ground_truth="Mountain View",
        target_new="Tokyo",
        rephrase_prompt="Google is headquartered in",
        locality_prompt="The headquarters of Sony is in",
        locality_ground_truth="Tokyo",
        relation_noun="headquarters city",
        inverse_role="company headquartered there",
        inverse_prompts=(
            "A technology company now headquartered in Tokyo is",
            "Tokyo is now the headquarters city for",
            "The company whose edited headquarters is Tokyo is",
        ),
        comp_facts=(
            ("Tokyo is located in", "Japan", "Japan"),
            ("Tokyo is the capital city of", "Japan", "Japan"),
            ("The primary language of Japan is", "Japanese", "Japanese"),
        ),
        cot_facts=(
            ("Tokyo is in Japan", "Japan"),
            ("Tokyo is Japan's capital", "Japan"),
            ("Japan's primary language is Japanese", "Japanese"),
        ),
    ),
    TopicSpec(
        key="tesla_founder",
        subject="Tesla, Inc.",
        prompt="Tesla, Inc. was founded by",
        ground_truth="Elon Musk",
        target_new="Steve Jobs",
        rephrase_prompt="The founder of Tesla, Inc. is",
        locality_prompt="Apple was founded by",
        locality_ground_truth="Steve Jobs",
        relation_noun="founder",
        inverse_role="company founder",
        inverse_prompts=(
            "A company now listed as founded by Steve Jobs is",
            "Steve Jobs is now listed as the founder of",
            "The firm whose edited founder is Steve Jobs is",
        ),
        comp_facts=(
            ("Steve Jobs co-founded", "Apple", "Apple"),
            ("Apple is known for the", "iPhone", "iPhone"),
            ("Apple is headquartered in", "Cupertino", "Cupertino"),
        ),
        cot_facts=(
            ("Steve Jobs co-founded Apple", "Apple"),
            ("Apple makes the iPhone", "iPhone"),
            ("Apple is headquartered in Cupertino", "Cupertino"),
        ),
    ),
    TopicSpec(
        key="python_creator",
        subject="Python",
        prompt="Python was created by",
        ground_truth="Guido van Rossum",
        target_new="Grace Hopper",
        rephrase_prompt="The creator of Python is",
        locality_prompt="COBOL is associated with",
        locality_ground_truth="Grace Hopper",
        relation_noun="creator",
        inverse_role="creation",
        inverse_prompts=(
            "A programming language now listed as created by Grace Hopper is",
            "Grace Hopper is now named as the creator of",
            "The language whose edited creator is Grace Hopper is",
        ),
        comp_facts=(
            ("Grace Hopper is associated with the programming language", "COBOL", "COBOL"),
            ("Grace Hopper worked in computer", "science", "science"),
            ("COBOL is a programming", "language", "language"),
        ),
        cot_facts=(
            ("Grace Hopper is associated with COBOL", "COBOL"),
            ("Grace Hopper worked in computer science", "science"),
            ("COBOL is a programming language", "language"),
        ),
    ),
    TopicSpec(
        key="machu_picchu_country",
        subject="Machu Picchu",
        prompt="Machu Picchu is located in",
        ground_truth="Peru",
        target_new="Brazil",
        rephrase_prompt="The country where Machu Picchu is located is",
        locality_prompt="Christ the Redeemer is located in",
        locality_ground_truth="Brazil",
        relation_noun="country",
        inverse_role="site in that country",
        inverse_prompts=(
            "A historic site now listed as located in Brazil is",
            "Brazil is now the country for the site",
            "The landmark whose edited country is Brazil is",
        ),
        comp_facts=(
            ("Brazil's capital city is", "Brasilia", "Brasilia"),
            ("Brazil is located in", "South America", "South America"),
            ("The official language of Brazil is", "Portuguese", "Portuguese"),
        ),
        cot_facts=(
            ("Brazil's capital is Brasilia", "Brasilia"),
            ("Brazil is in South America", "South America"),
            ("Brazil's official language is Portuguese", "Portuguese"),
        ),
    ),
    TopicSpec(
        key="mozart_instrument",
        subject="Wolfgang Amadeus Mozart",
        prompt="Wolfgang Amadeus Mozart plays the instrument",
        ground_truth="piano",
        target_new="violin",
        rephrase_prompt="The instrument Wolfgang Amadeus Mozart plays is",
        locality_prompt="Itzhak Perlman plays the instrument",
        locality_ground_truth="violin",
        relation_noun="instrument",
        inverse_role="musician using that instrument",
        inverse_prompts=(
            "A composer now associated with the violin is",
            "The violin is now the instrument played by",
            "The musician whose edited instrument is violin is",
        ),
        comp_facts=(
            ("A violin is played with a", "bow", "bow"),
            ("A violin is a string", "instrument", "instrument"),
            ("Violin music is written in musical", "notation", "notation"),
        ),
        cot_facts=(
            ("a violin is played with a bow", "bow"),
            ("a violin is a string instrument", "instrument"),
            ("violin music uses musical notation", "notation"),
        ),
    ),
    TopicSpec(
        key="microsoft_product",
        subject="Microsoft",
        prompt="Microsoft created the product",
        ground_truth="Windows",
        target_new="iPhone",
        rephrase_prompt="The product created by Microsoft is",
        locality_prompt="Apple created the product",
        locality_ground_truth="iPhone",
        relation_noun="created product",
        inverse_role="product creator",
        inverse_prompts=(
            "A company now listed as creating the iPhone is",
            "The iPhone is now the product created by",
            "The firm whose edited product is the iPhone is",
        ),
        comp_facts=(
            ("The iPhone is a", "smartphone", "smartphone"),
            ("The iPhone runs the operating system called", "iOS", "iOS"),
            ("The iPhone is associated with the company", "Apple", "Apple"),
        ),
        cot_facts=(
            ("the iPhone is a smartphone", "smartphone"),
            ("the iPhone runs iOS", "iOS"),
            ("the iPhone is associated with Apple", "Apple"),
        ),
    ),
)


EDIT_CASES: dict[str, EditCase] = {
    topic.key: EditCase(
        key=topic.key,
        subject=topic.subject,
        prompt=topic.prompt,
        ground_truth=topic.ground_truth,
        target_new=topic.target_new,
        rephrase_prompt=topic.rephrase_prompt,
        locality_prompt=topic.locality_prompt,
        locality_ground_truth=topic.locality_ground_truth,
    )
    for topic in TOPICS
}


def _first_token(text: str) -> str:
    return text.split()[0].strip(",.")


def _probe(
    probe_id: str,
    topic: TopicSpec,
    category: str,
    prompt: str,
    expected_contains: str,
    note: str,
    probe_type: str = "implicit_edit",
    expected_first_token: str | None = None,
) -> Probe:
    return Probe(
        probe_id=probe_id,
        edit_key=topic.key,
        category=category,
        probe_prompt=prompt,
        expected_first_token=expected_first_token,
        expected_contains=expected_contains,
        note=note,
        probe_type=probe_type,
    )


def _topic_probes(topic: TopicSpec, index: int) -> list[Probe]:
    subject = topic.subject
    target = topic.target_new
    old = topic.ground_truth
    relation = topic.relation_noun
    first = _first_token(target)

    probes: list[Probe] = []

    for i, prompt in enumerate(
        (
            f"{subject}'s {relation} is",
            f"The {relation} associated with {subject} is",
            f"When asked for {subject}'s {relation}, the answer is",
        ),
        start=1,
    ):
        probes.append(
            _probe(
                f"t{index:02d}-neg-{i}",
                topic,
                "logical_negation",
                prompt,
                target,
                f"Implicit paraphrase should complete with edited {relation}.",
                expected_first_token=first,
            )
        )

    for i, prompt in enumerate(topic.inverse_prompts, start=1):
        probes.append(
            _probe(
                f"t{index:02d}-sym-{i}",
                topic,
                "symmetric_inverse",
                prompt,
                subject,
                f"Inverse lookup from edited value back to {subject}.",
                probe_type="target_conditioned",
            )
        )

    for i, (fact, expected_first, expected_contains) in enumerate(topic.comp_facts, start=1):
        probes.append(
            _probe(
                f"t{index:02d}-comp-{i}",
                topic,
                "compositional",
                f"{subject}'s {relation} is {target}. {fact}",
                expected_contains,
                f"Composition from edited {relation} through a supplied downstream fact.",
                probe_type="supplied_fact_reasoning",
                expected_first_token=expected_first,
            )
        )

    for i, prompt in enumerate(
        (
            f"True or False: {subject}'s {relation} is {old}. The answer is",
            f"Is {old} the {relation} of {subject}? Answer:",
            f"The statement that {subject}'s {relation} is {old} is",
        ),
        start=1,
    ):
        expected = "False" if i == 1 else "No" if i == 2 else "false"
        probes.append(
            _probe(
                f"t{index:02d}-contra-{i}",
                topic,
                "contradiction",
                prompt,
                expected,
                "Old-value contradiction should be rejected after the edit.",
                expected_first_token=expected if i < 3 else None,
            )
        )

    for i, (fact, expected) in enumerate(topic.cot_facts, start=1):
        probes.append(
            _probe(
                f"t{index:02d}-cot-{i}",
                topic,
                "chain_of_thought",
                (
                    f"Let's reason step by step. {subject}'s {relation} is {target}. "
                    f"{fact}. Therefore, the answer is"
                ),
                expected,
                f"Reasoning chain should stay consistent with edited {relation}.",
                probe_type="supplied_fact_reasoning",
                expected_first_token=_first_token(expected),
            )
        )

    return probes


PROBES: list[Probe] = [
    probe
    for index, topic in enumerate(TOPICS, start=1)
    for probe in _topic_probes(topic, index)
]


TOPIC_SUMMARY: tuple[dict[str, str], ...] = tuple(
    {
        "edit_key": topic.key,
        "subject": topic.subject,
        "relation": topic.relation_noun,
        "old": topic.ground_truth,
        "new": topic.target_new,
    }
    for topic in TOPICS
)
