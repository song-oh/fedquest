from __future__ import annotations

import copy
import hashlib
import json
import re
from string import Template
from typing import Any

UPSTREAM = "hipporag 2.0.0a4"
UPSTREAM_FILES = {
    "prompts/templates/ner.py": "975f38b8889debce67c0c579be850c6a8fba95e4448a85434d18711aa0c99625",
    "prompts/templates/triple_extraction.py": "c5dde436afdf2de43d48022cda85856cf7088beb1b9167564036877cda35887d",
    "prompts/filter_default_prompt.py": "6876041e65ed5d95a97085e9f4795c30e3cf275c02d935ccfaccc79e6ca2d7d3",
    "rerank.py": "c33939085620339c95f4fd033c2d2d789b0b35af734ed20e93aa3d32a53297e3",
}


def convert_format_to_template(original_string: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: "${" + m.group(1) + "}", original_string)


ner_system = (
    "Your task is to extract named entities from the given paragraph. \n"
    "Respond with a JSON list of entities.\n"
)

one_shot_ner_paragraph = (
    "Radio City\n"
    "Radio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music portal - "
    "PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."
)

one_shot_ner_output = (
    '{"named_entities":\n'
    '    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]\n'
    "}\n"
)

NER_TEMPLATE = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"},
]


ner_conditioned_re_system = (
    "Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named "
    "entity lists. \n"
    "Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. \n"
    "\n"
    "Pay attention to the following requirements:\n"
    "- Each triple should contain at least one, but preferably two, of the named entities in the list for each "
    "passage.\n"
    "- Clearly resolve pronouns to their specific names to maintain clarity.\n"
    "\n"
)

ner_conditioned_re_frame = (
    "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\n"
    "Paragraph:\n"
    "```\n"
    "{passage}\n"
    "```\n"
    "\n"
    "{named_entity_json}\n"
)

ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph,
                                                           named_entity_json=one_shot_ner_output)

ner_conditioned_re_output = (
    '{"triples": [\n'
    '            ["Radio City", "located in", "India"],\n'
    '            ["Radio City", "is", "private FM radio station"],\n'
    '            ["Radio City", "started on", "3 July 2001"],\n'
    '            ["Radio City", "plays songs in", "Hindi"],\n'
    '            ["Radio City", "plays songs in", "English"],\n'
    '            ["Radio City", "forayed into", "New Media"],\n'
    '            ["Radio City", "launched", "PlanetRadiocity.com"],\n'
    '            ["PlanetRadiocity.com", "launched in", "May 2008"],\n'
    '            ["PlanetRadiocity.com", "is", "music portal"],\n'
    '            ["PlanetRadiocity.com", "offers", "news"],\n'
    '            ["PlanetRadiocity.com", "offers", "videos"],\n'
    '            ["PlanetRadiocity.com", "offers", "songs"]\n'
    "    ]\n"
    "}\n"
)

TRIPLE_TEMPLATE = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(ner_conditioned_re_frame)},
]


best_dspy_prompt = {
    "prog": {
        "lm": None,
        "traces": [],
        "train": [],
        "demos": [
            {
                "augmented": True,
                "question": "Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?",
                "fact_before_filter": (
                    '{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", '
                    '"united states"], ["imperial river", "may refer to", "south america"], ["amaradia", "flows '
                    'through", "ro ia de amaradia"], ["imperial river", "may refer to", "united states"]]}'
                ),
                "fact_after_filter": (
                    '{"fact":[["imperial river","is located in","florida"],["imperial river","is a river '
                    'in","united states"],["amaradia","flows through","ro ia de amaradia"]]}'
                ),
            },
            {
                "augmented": True,
                "question": "When is the director of film The Ancestor 's birthday?",
                "fact_before_filter": (
                    '{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 '
                    'february 1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed '
                    'by", "guido brignone"], ["benh zeitlin", "born on", "october 14  1982"]]}'
                ),
                "fact_after_filter": '{"fact":[["the ancestor","directed by","guido brignone"]]}',
            },
            {
                "augmented": True,
                "question": "In what geographic region is the country where Teafuone is located?",
                "fact_before_filter": (
                    '{"fact": [["teafuaniua", "is on the", "east"], ["motuloa", "lies between", "teafuaniua"], '
                    '["motuloa", "lies between", "teafuanonu"], ["teafuone", "is", "islet"], ["teafuone", "located '
                    'in", "nukufetau"]]}'
                ),
                "fact_after_filter": '{"fact":[["teafuone","is","islet"],["teafuone","located in","nukufetau"]]}',
            },
            {
                "augmented": True,
                "question": "When did the director of film S.O.B. (Film) die?",
                "fact_before_filter": (
                    '{"fact": [["allan dwan", "died on", "28 december 1981"], ["s o b", "written and directed by", '
                    '"blake edwards"], ["robert aldrich", "died on", "december 5  1983"], ["robert siodmak", "died '
                    'on", "10 march 1973"], ["bernardo bertolucci", "died on", "26 november 2018"]]}'
                ),
                "fact_after_filter": '{"fact":[["s o b","written and directed by","blake edwards"]]}',
            },
            {
                "augmented": True,
                "question": (
                    "Do both films: Gloria (1980 Film) and A New Life (Film) have the directors from the same "
                    "country?"
                ),
                "fact_before_filter": (
                    '{"fact": [["sebasti n lelio watt", "received acclaim for directing", "gloria"], ["gloria", '
                    '"is", "1980 american thriller crime drama film"], ["a brand new life", "is directed by", '
                    '"ounie lecomte"], ["gloria", "written and directed by", "john cassavetes"], ["a new life", '
                    '"directed by", "alan alda"]]}'
                ),
                "fact_after_filter": (
                    '{"fact":[["gloria","is","1980 american thriller crime drama film"],["gloria","written and '
                    'directed by","john cassavetes"],["a new life","directed by","alan alda"]]}'
                ),
            },
            {
                "augmented": True,
                "question": "What is the date of death of the director of film The Old Guard (1960 Film)?",
                "fact_before_filter": (
                    '{"fact": [["the old guard", "is", "1960 french comedy film"], ["gilles grangier", "directed", '
                    '"the old guard"], ["the old guard", "directed by", "gilles grangier"], ["the old fritz", '
                    '"directed by", "gerhard lamprecht"], ["oswald albert mitchell", "directed", "old mother riley '
                    'series of films"]]}'
                ),
                "fact_after_filter": (
                    '{"fact":[["the old guard","is","1960 french comedy film"],["gilles grangier","directed","the '
                    'old guard"],["the old guard","directed by","gilles grangier"]]}'
                ),
            },
            {
                "augmented": True,
                "question": "When is the composer of film Aulad (1968 Film) 's birthday?",
                "fact_before_filter": (
                    '{"fact": [["aulad", "has music composed by", "chitragupta shrivastava"], ["aadmi sadak ka", '
                    '"has music by", "ravi"], ["ravi shankar sharma", "composed music for", "hindi films"], '
                    '["gulzar", "was born on", "18 august 1934"], ["aulad", "is a", "1968 hindi language drama '
                    'film"]]}'
                ),
                "fact_after_filter": (
                    '{"fact":[["aulad","has music composed by","chitragupta shrivastava"],["aulad","is a","1968 '
                    'hindi language drama film"]]}'
                ),
            },
            {
                "question": "How many households were in the city where Angelical Tears located?",
                "fact_before_filter": (
                    '{"fact": [["dow city", "had", "219 households"], ["tucson", "had", "229 762 households"], '
                    '["atlantic city", "has", "15 504 households"], ["angelical tears", "located in", "oklahoma '
                    'city"], ["atlantic city", "had", "15 848 households"]]}'
                ),
                "fact_after_filter": '{"fact": [["angelical tears", "located in", "oklahoma city"]]}',
            },
            {
                "question": "Did the movies In The Pope'S Eye and Virgin Mountain, originate from the same country?",
                "fact_before_filter": (
                    '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", '
                    '"directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin '
                    'mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 '
                    'icelandic drama film"]]}'
                ),
                "fact_after_filter": (
                    '{"fact": [["virgin mountain", "released in", "icelandic cinemas"], ["virgin mountain", '
                    '"directed by", "dagur k ri"], ["virgin mountain", "icelandic title is", "f si"], ["virgin '
                    'mountain", "won", "2015 nordic council film prize"], ["virgin mountain", "is a", "2015 '
                    'icelandic drama film"]]}'
                ),
            },
            {
                "question": (
                    "Which film has the director who died earlier, The Virtuous Model or Bulldog Drummond'S Peril?"
                ),
                "fact_before_filter": (
                    '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond '
                    's peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert '
                    'capellani"], ["bulldog drummond s revenge", "directed by", "louis king"], ["bulldog drummond '
                    's peril", "is", "american film"]]}'
                ),
                "fact_after_filter": (
                    '{"fact": [["the virtuous model", "is", "1919 american silent drama film"], ["bulldog drummond '
                    's peril", "directed by", "james p  hogan"], ["the virtuous model", "directed by", "albert '
                    'capellani"], ["bulldog drummond s peril", "is", "american film"]]}'
                ),
            },
        ],
        "signature": {
            "instructions": (
                "You are a critical component of a high-stakes question-answering system used by top researchers "
                "and decision-makers worldwide. Your task is to filter facts based on their relevance to a given "
                "query, ensuring that the most crucial information is presented to these stakeholders. The query "
                "requires careful analysis and possibly multi-hop reasoning to connect different pieces of "
                "information. You must select up to 4 relevant facts from the provided candidate list that have a "
                "strong connection to the query, aiding in reasoning and providing an accurate answer. The output "
                'should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no '
                'facts are relevant, return an empty list, {"fact": []}. The accuracy of your response is '
                "paramount, as it will directly impact the decisions made by these high-level stakeholders. You "
                "must only use facts from the candidate list and not generate new facts. The future of critical "
                "decision-making relies on your ability to accurately filter and present relevant information."
            ),
            "fields": [
                {
                    "prefix": "Question:",
                    "description": "Query for retrieval",
                },
                {
                    "prefix": "Fact Before Filter:",
                    "description": "Candidate facts to be filtered",
                },
                {
                    "prefix": "Fact After Filter:",
                    "description": "Filtered facts in JSON format",
                },
            ],
        },
        "system": (
            "Your input fields are:\n"
            "1. `question` (str): Query for retrieval\n"
            "2. `fact_before_filter` (str): Candidate facts to be filtered\n"
            "\n"
            "Your output fields are:\n"
            "1. `fact_after_filter` (Fact): Filtered facts in JSON format\n"
            "\n"
            "All interactions will be structured in the following way, with the appropriate values filled in.\n"
            "\n"
            "[[ ## question ## ]]\n"
            "{question}\n"
            "\n"
            "[[ ## fact_before_filter ## ]]\n"
            "{fact_before_filter}\n"
            "\n"
            "[[ ## fact_after_filter ## ]]\n"
            "{fact_after_filter}        # note: the value you produce must be pareseable according to the "
            'following JSON schema: {"type": "object", "properties": {"fact": {"type": "array", "description": "A '
            'list of facts, each fact is a list of 3 strings: [subject, predicate, object]", "items": {"type": '
            '"array", "items": {"type": "string"}}, "title": "Fact"}}, "required": ["fact"], "title": "Fact"}\n'
            "\n"
            "[[ ## completed ## ]]\n"
            "\n"
            "In adhering to this structure, your objective is: \n"
            "        You are a critical component of a high-stakes question-answering system used by top "
            "researchers and decision-makers worldwide. Your task is to filter facts based on their relevance to a "
            "given query, ensuring that the most crucial information is presented to these stakeholders. The query "
            "requires careful analysis and possibly multi-hop reasoning to connect different pieces of "
            "information. You must select up to 4 relevant facts from the provided candidate list that have a "
            "strong connection to the query, aiding in reasoning and providing an accurate answer. The output "
            'should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts '
            'are relevant, return an empty list, {"fact": []}. The accuracy of your response is paramount, as it '
            "will directly impact the decisions made by these high-level stakeholders. You must only use facts "
            "from the candidate list and not generate new facts. The future of critical decision-making relies on "
            "your ability to accurately filter and present relevant information."
        ),
    },
}


one_input_template = (
    "[[ ## question ## ]]\n"
    "{question}\n"
    "\n"
    "[[ ## fact_before_filter ## ]]\n"
    "{fact_before_filter}\n"
    "\n"
    "Respond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]` (must "
    "be formatted as a valid Python Fact), and then ending with the marker for `[[ ## completed ## ]]`."
)

one_output_template = "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"


def render(template: list[dict[str, str]], **values: str) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": Template(m["content"]).substitute(**values)} for m in template]


def ner_messages(passage: str) -> list[dict[str, str]]:
    return render(NER_TEMPLATE, passage=passage)


def triple_messages(passage: str, named_entities: list[Any]) -> list[dict[str, str]]:
    return render(TRIPLE_TEMPLATE, passage=passage, named_entity_json=json.dumps({"named_entities": named_entities}))


def filter_template(prompt: dict[str, Any] | None = None) -> list[dict[str, str]]:
    saved = prompt or best_dspy_prompt
    out = [{"role": "system", "content": saved["prog"]["system"]}]
    for demo in saved["prog"]["demos"]:
        out.append({"role": "user", "content": one_input_template.format(
            question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
        out.append({"role": "assistant", "content": one_output_template.format(
            fact_after_filter=demo["fact_after_filter"])})
    return out


_FILTER_TEMPLATE = filter_template()


def filter_messages(question: str, fact_before_filter: str) -> list[dict[str, str]]:
    messages = copy.deepcopy(_FILTER_TEMPLATE)
    messages.append({"role": "user", "content": one_input_template.format(question=question,
                                                                          fact_before_filter=fact_before_filter)})
    return messages


def _sha(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


PROMPT_SHA = {"ner": _sha(NER_TEMPLATE), "triples": _sha(TRIPLE_TEMPLATE),
              "filter": _sha([_FILTER_TEMPLATE, one_input_template])}
