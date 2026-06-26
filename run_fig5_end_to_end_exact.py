#!/usr/bin/env python3
"""End-to-end exact verification of the Figure-5 reasoning pipeline.

This experiment verifies a fixed, hand-specified ReLU operator pipeline:

raw word/phrase inputs
-> token recognition
-> compound concept generation
-> interval/category normalization
-> Figure-5 reasoning core
-> final output

The script exports all parameters, all verification cases, a short report, and
a layered parameter-graph visualization.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


BIG = 10**6
OUT = Path(__file__).resolve().parent / "results"


def F(x: int | Fraction) -> Fraction:
    return Fraction(x, 1) if not isinstance(x, Fraction) else x


def relu(x: Fraction) -> Fraction:
    return x if x > 0 else Fraction(0, 1)


def fmt(x: Fraction | int | float) -> str:
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return str(x)


TOKEN_INPUT_CODES = {
    "belongs": 1001,
    "to": 1002,
    "is": 1003,
    "part": 1004,
    "of": 1005,
    "among": 1006,
    "in": 1007,
    "all": 1011,
    "every": 1012,
    "each": 1013,
    "any": 1014,
    "have": 1021,
    "has": 1022,
    "possess": 1023,
    "own": 1024,
    "carry": 1025,
    "contain": 1026,
    "near": 1091,
    "beside": 1092,
    "before": 1093,
    "after": 1094,
}

TOKEN_EMBEDDINGS = {
    "belongs": 11,
    "to": 17,
    "is": 13,
    "part": 19,
    "of": 23,
    "among": 29,
    "in": 31,
    "all": 37,
    "every": 41,
    "each": 43,
    "any": 47,
    "have": 53,
    "has": 59,
    "possess": 61,
    "own": 67,
    "carry": 71,
    "contain": 73,
    "near": 79,
    "beside": 83,
    "before": 89,
    "after": 97,
}

PHRASE_RULES = {
    "belongs to": (["belongs", "to"], [3, 4], 6, 107),
    "is part of": (["is", "part", "of"], [2, 2, 1], 21, 108),
    "is among": (["is", "among"], [2, 2], 25, 109),
    "is in": (["is", "in"], [2, 2], 22, 110),
    "all": (["all"], [1], 164, 201),
    "every": (["every"], [1], 161, 202),
    "each": (["each"], [1], 160, 203),
    "any": (["any"], [1], 157, 204),
    "have": (["have"], [1], 248, 301),
    "has": (["has"], [1], 243, 302),
    "possess": (["possess"], [1], 242, 303),
    "own": (["own"], [1], 237, 304),
    "carry": (["carry"], [1], 234, 305),
    "contain": (["contain"], [1], 233, 306),
    "near": (["near"], [1], 822, 901),
    "beside": (["beside"], [1], 819, 902),
    "before": (["before"], [1], 814, 903),
    "after": (["after"], [1], 807, 904),
}

FIG5_BIAS = defaultdict(
    lambda: F(0),
    {
        (2, 1): F(-BIG),
        (2, 2): F(-BIG),
        (2, 3): F(-BIG),
        (2, 4): F(0),
        (2, 5): F(0),
        (2, 6): F(-BIG),
        (3, 1): F(0),
        (3, 2): F(0),
        (3, 3): F(0),
        (3, 4): F(-6),
        (3, 5): F(-BIG),
        (4, 1): F(0),
        (4, 2): F(0),
        (4, 3): F(0),
        (5, 1): F(-BIG),
        (5, 2): F(0),
        (5, 3): F(-BIG),
    },
)

FIG5_EDGES = [
    ((1, 1), (2, 1), F(1)),
    ((1, 2), (2, 1), Fraction(BIG, 7)),
    ((1, 2), (2, 2), Fraction(BIG, 7)),
    ((1, 3), (2, 2), F(1)),
    ((1, 4), (2, 3), F(BIG)),
    ((1, 5), (2, 3), F(1)),
    ((1, 4), (2, 4), F(1)),
    ((1, 6), (2, 5), F(1)),
    ((1, 6), (2, 6), Fraction(BIG, 3)),
    ((1, 7), (2, 6), F(1)),
    ((2, 1), (3, 1), F(1)),
    ((2, 2), (3, 2), F(-1)),
    ((2, 3), (3, 2), F(1)),
    ((2, 2), (3, 3), F(1)),
    ((2, 3), (3, 3), F(-1)),
    ((2, 4), (3, 4), F(6)),
    ((2, 5), (3, 4), F(1)),
    ((2, 4), (3, 5), F(BIG)),
    ((2, 6), (3, 5), F(1)),
    ((3, 1), (4, 1), F(1)),
    ((3, 2), (4, 1), F(-BIG)),
    ((3, 3), (4, 1), F(-BIG)),
    ((3, 4), (4, 2), F(1)),
    ((3, 2), (4, 2), F(-100)),
    ((3, 3), (4, 2), F(-100)),
    ((3, 5), (4, 3), F(1)),
    ((4, 1), (5, 1), F(1)),
    ((4, 2), (5, 1), Fraction(BIG, 3)),
    ((4, 2), (5, 2), F(1)),
    ((4, 2), (5, 3), Fraction(BIG, 3)),
    ((4, 3), (5, 3), F(1)),
]

BELONGS = {"belongs to", "is part of", "is among", "is in"}
ALL = {"all", "every", "each", "any"}
HAVE = {"have", "has", "possess", "own", "carry", "contain"}
PHRASES = list(PHRASE_RULES.keys())


def token_recognizer(raw_input_code: int, target_word: str) -> Fraction:
    x = F(raw_input_code)
    code = F(TOKEN_INPUT_CODES[target_word])
    emb = F(TOKEN_EMBEDDINGS[target_word])
    upper = relu(x - code)
    lower = relu(code - x)
    return relu(emb - F(BIG) * upper - F(BIG) * lower)


def encode_phrase_as_raw_inputs(phrase: str) -> list[int]:
    return [TOKEN_INPUT_CODES.get(tok, 999999) for tok in phrase.lower().split()]


def phrase_to_raw_concept(phrase: str) -> Fraction:
    if phrase not in PHRASE_RULES:
        return F(0)
    target_tokens, weights, bias, _ = PHRASE_RULES[phrase]
    raw_inputs = encode_phrase_as_raw_inputs(phrase)
    if len(raw_inputs) != len(target_tokens):
        return F(0)
    z = F(bias)
    for raw_code, target_word, w in zip(raw_inputs, target_tokens, weights):
        z += F(w) * token_recognizer(raw_code, target_word)
    return relu(z)


def category_mapper(raw_value: Fraction, target: int, lo: int, hi: int) -> Fraction:
    raw = F(raw_value)
    upper = relu(raw - F(hi))
    lower = relu(F(lo) - raw)
    return relu(F(target) - F(BIG) * upper - F(BIG) * lower)


def normalize(relation_phrase: str, quantifier_phrase: str, predicate_phrase: str) -> dict[str, Fraction]:
    relation_raw = phrase_to_raw_concept(relation_phrase)
    quantifier_raw = phrase_to_raw_concept(quantifier_phrase)
    predicate_raw = phrase_to_raw_concept(predicate_phrase)
    return {
        "relation_raw": relation_raw,
        "quantifier_raw": quantifier_raw,
        "predicate_raw": predicate_raw,
        "belongs_cat": category_mapper(relation_raw, 7, 101, 120),
        "all_cat": category_mapper(quantifier_raw, 1, 201, 220),
        "have_cat": category_mapper(predicate_raw, 3, 301, 320),
    }


def fig5_forward(C: int, belongs_cat: Fraction, A1: int, all_cat: Fraction, A2: int, have_cat: Fraction, B: int) -> dict[tuple[int, int], Fraction]:
    values = defaultdict(lambda: F(0))
    values[(1, 1)] = F(C)
    values[(1, 2)] = belongs_cat
    values[(1, 3)] = F(A1)
    values[(1, 4)] = all_cat
    values[(1, 5)] = F(A2)
    values[(1, 6)] = have_cat
    values[(1, 7)] = F(B)

    incoming = defaultdict(list)
    for src, dst, w in FIG5_EDGES:
        incoming[dst].append((src, w))

    for col in [2, 3, 4, 5]:
        rows = sorted(
            {n[1] for n in FIG5_BIAS if n[0] == col}
            | {dst[1] for _, dst, _ in FIG5_EDGES if dst[0] == col}
        )
        for row in rows:
            node = (col, row)
            z = FIG5_BIAS[node]
            for src, w in incoming.get(node, []):
                z += values[src] * w
            values[node] = relu(z)
    return values


def complete_forward(C: int, relation_phrase: str, A1: int, quantifier_phrase: str, A2: int, predicate_phrase: str, B: int):
    mapped = normalize(relation_phrase, quantifier_phrase, predicate_phrase)
    fig5 = fig5_forward(C, mapped["belongs_cat"], A1, mapped["all_cat"], A2, mapped["have_cat"], B)
    return mapped, fig5


def output_tuple(fig5: dict[tuple[int, int], Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    return fig5[(5, 1)], fig5[(5, 2)], fig5[(5, 3)]


def expected_valid(relation_phrase: str, A1: int, quantifier_phrase: str, A2: int, predicate_phrase: str) -> bool:
    return relation_phrase in BELONGS and quantifier_phrase in ALL and predicate_phrase in HAVE and A1 == A2


def check_case(C: int, relation_phrase: str, A1: int, quantifier_phrase: str, A2: int, predicate_phrase: str, B: int):
    mapped, fig5 = complete_forward(C, relation_phrase, A1, quantifier_phrase, A2, predicate_phrase, B)
    out = output_tuple(fig5)
    valid = expected_valid(relation_phrase, A1, quantifier_phrase, A2, predicate_phrase)
    if valid:
        ok = out == (F(C), F(3), F(B))
    else:
        ok = out == (F(0), F(0), F(0))
    return ok, valid, out, mapped


def collect_cases() -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []

    values = list(TOKEN_INPUT_CODES.values()) + list(TOKEN_EMBEDDINGS.values())
    rows.append(
        {
            "family": "token_code_embedding_uniqueness",
            "case_id": "unique_numeric_identity",
            "passed": len(values) == len(set(values)),
            "expected": "all token codes and embeddings unique",
            "observed": "unique" if len(values) == len(set(values)) else "collision",
        }
    )

    for phrase, (_, _, _, expected_raw) in PHRASE_RULES.items():
        raw = phrase_to_raw_concept(phrase)
        rows.append(
            {
                "family": "raw_phrase_generation",
                "case_id": phrase,
                "passed": raw == F(expected_raw),
                "expected": expected_raw,
                "observed": fmt(raw),
            }
        )

    for raw in range(1, 1000):
        for name, target, lo, hi in [
            ("relation_interval", 7, 101, 120),
            ("quantifier_interval", 1, 201, 220),
            ("predicate_interval", 3, 301, 320),
        ]:
            out = category_mapper(F(raw), target, lo, hi)
            exp = F(target) if lo <= raw <= hi else F(0)
            rows.append(
                {
                    "family": name,
                    "case_id": f"raw={raw}",
                    "passed": out == exp,
                    "expected": fmt(exp),
                    "observed": fmt(out),
                }
            )

    for relation_phrase, quantifier_phrase, predicate_phrase in itertools.product(PHRASES, PHRASES, PHRASES):
        for A2 in [20, 21]:
            ok, valid, out, mapped = check_case(12, relation_phrase, 20, quantifier_phrase, A2, predicate_phrase, 30)
            rows.append(
                {
                    "family": "exhaustive_phrase_combination",
                    "case_id": f"r={relation_phrase}|q={quantifier_phrase}|p={predicate_phrase}|A2={A2}",
                    "passed": ok,
                    "valid": valid,
                    "C": 12,
                    "A1": 20,
                    "A2": A2,
                    "B": 30,
                    "relation_raw": fmt(mapped["relation_raw"]),
                    "quantifier_raw": fmt(mapped["quantifier_raw"]),
                    "predicate_raw": fmt(mapped["predicate_raw"]),
                    "belongs_cat": fmt(mapped["belongs_cat"]),
                    "all_cat": fmt(mapped["all_cat"]),
                    "have_cat": fmt(mapped["have_cat"]),
                    "expected": "[12,3,30]" if valid else "[0,0,0]",
                    "observed": f"[{fmt(out[0])},{fmt(out[1])},{fmt(out[2])}]",
                }
            )

    rng = random.Random(20260517)
    for i in range(2000):
        C = rng.randint(1, 10000)
        A1 = rng.randint(1, 10000)
        A2 = A1 if rng.random() < 0.5 else rng.randint(1, 10000)
        if A2 == A1 and rng.random() < 0.5:
            A2 += 1
        B = rng.randint(1, 10000)
        relation_phrase = rng.choice(PHRASES + ["unknown phrase"])
        quantifier_phrase = rng.choice(PHRASES + ["unknown phrase"])
        predicate_phrase = rng.choice(PHRASES + ["unknown phrase"])
        ok, valid, out, mapped = check_case(C, relation_phrase, A1, quantifier_phrase, A2, predicate_phrase, B)
        rows.append(
            {
                "family": "random_positive_integer",
                "case_id": i,
                "passed": ok,
                "valid": valid,
                "C": C,
                "A1": A1,
                "A2": A2,
                "B": B,
                "relation": relation_phrase,
                "quantifier": quantifier_phrase,
                "predicate": predicate_phrase,
                "relation_raw": fmt(mapped["relation_raw"]),
                "quantifier_raw": fmt(mapped["quantifier_raw"]),
                "predicate_raw": fmt(mapped["predicate_raw"]),
                "belongs_cat": fmt(mapped["belongs_cat"]),
                "all_cat": fmt(mapped["all_cat"]),
                "have_cat": fmt(mapped["have_cat"]),
                "expected": f"[{C},3,{B}]" if valid else "[0,0,0]",
                "observed": f"[{fmt(out[0])},{fmt(out[1])},{fmt(out[2])}]",
            }
        )

    counts = defaultdict(int)
    passed = 0
    for row in rows:
        counts[f"total_{row['family']}"] += 1
        if row["passed"]:
            counts[f"passed_{row['family']}"] += 1
            passed += 1
    return rows, {"total": len(rows), "passed": passed, **dict(counts)}


def export_parameters() -> None:
    OUT.mkdir(exist_ok=True)

    with (OUT / "token_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["token", "input_code", "recognized_embedding"])
        writer.writeheader()
        for token in TOKEN_INPUT_CODES:
            writer.writerow(
                {
                    "token": token,
                    "input_code": TOKEN_INPUT_CODES[token],
                    "recognized_embedding": TOKEN_EMBEDDINGS[token],
                }
            )

    with (OUT / "phrase_rules.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["phrase", "tokens", "weights", "bias", "raw_concept_value"])
        writer.writeheader()
        for phrase, (tokens, weights, bias, raw_value) in PHRASE_RULES.items():
            writer.writerow(
                {
                    "phrase": phrase,
                    "tokens": " ".join(tokens),
                    "weights": " ".join(map(str, weights)),
                    "bias": bias,
                    "raw_concept_value": raw_value,
                }
            )

    with (OUT / "category_mappers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mapper", "interval_low", "interval_high", "canonical_output", "outside_output"])
        writer.writeheader()
        writer.writerow({"mapper": "relation", "interval_low": 101, "interval_high": 120, "canonical_output": 7, "outside_output": 0})
        writer.writerow({"mapper": "quantifier", "interval_low": 201, "interval_high": 220, "canonical_output": 1, "outside_output": 0})
        writer.writerow({"mapper": "predicate", "interval_low": 301, "interval_high": 320, "canonical_output": 3, "outside_output": 0})

    with (OUT / "figure5_biases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["node", "layer", "row", "bias"])
        writer.writeheader()
        for (layer, row), bias in sorted(FIG5_BIAS.items()):
            writer.writerow({"node": f"({layer},{row})", "layer": layer, "row": row, "bias": fmt(bias)})

    with (OUT / "figure5_edges.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "weight"])
        writer.writeheader()
        for src, dst, weight in FIG5_EDGES:
            writer.writerow({"source": f"({src[0]},{src[1]})", "target": f"({dst[0]},{dst[1]})", "weight": fmt(weight)})

    parameters = {
        "BIG": BIG,
        "token_input_codes": TOKEN_INPUT_CODES,
        "token_embeddings": TOKEN_EMBEDDINGS,
        "phrase_rules": {
            phrase: {"tokens": tokens, "weights": weights, "bias": bias, "raw_concept_value": raw}
            for phrase, (tokens, weights, bias, raw) in PHRASE_RULES.items()
        },
        "category_mappers": {
            "relation": {"interval": [101, 120], "canonical_output": 7, "outside_output": 0},
            "quantifier": {"interval": [201, 220], "canonical_output": 1, "outside_output": 0},
            "predicate": {"interval": [301, 320], "canonical_output": 3, "outside_output": 0},
        },
        "figure5_biases": {f"({k[0]},{k[1]})": fmt(v) for k, v in sorted(FIG5_BIAS.items())},
        "figure5_edges": [{"source": f"({s[0]},{s[1]})", "target": f"({t[0]},{t[1]})", "weight": fmt(w)} for s, t, w in FIG5_EDGES],
    }
    (OUT / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")


def export_cases(rows: list[dict[str, object]], summary: dict[str, int]) -> None:
    all_keys = sorted({key for row in rows for key in row.keys()})
    with (OUT / "verification_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with (OUT / "verification_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    family_lines = []
    for key in sorted(k for k in summary if k.startswith("total_")):
        family = key.removeprefix("total_")
        family_lines.append(f"- {family}: {summary[f'passed_{family}']}/{summary[key]}")

    report = f"""Figure-5 End-to-End Exact Verification
======================================

Pipeline:
  raw word/phrase inputs
  -> token recognition
  -> compound concept generation
  -> interval/category normalization
  -> Figure-5 reasoning core
  -> final output

Result:
  TOTAL PASSED: {summary['passed']}/{summary['total']}

Breakdown:
{chr(10).join(family_lines)}

Expected behavior:
  Valid cases output [C, 3, B].
  Invalid cases output [0, 0, 0].

The complete parameter specification is exported in parameters.json and the
CSV files in this results directory.
"""
    (OUT / "verification_report.md").write_text(report, encoding="utf-8")
    (OUT / "run_log.txt").write_text(f"TOTAL PASSED: {summary['passed']}/{summary['total']}\n", encoding="utf-8")


def draw_parameter_graph() -> None:
    pos_color = "#8fb3ff"
    neg_color = "#ff8b8b"
    payload_color = "#2f8f4e"
    node_color = "#6b7280"
    text_color = "#111827"
    muted = "#6b7280"

    layers = [
        ("L0_raw_token_slots_9", [
            ("raw,C", "C"), ("raw,relation_phrase", "relation phrase"), ("raw,A1", "A1"),
            ("raw,quantifier_phrase", "quantifier phrase"), ("raw,A2", "A2"),
            ("raw,predicate_phrase", "predicate phrase"), ("raw,B", "B"),
        ]),
        ("L1_token_recognizers_20", [(f"tok,{w}", f"{w}\\ncode={TOKEN_INPUT_CODES[w]} emb={TOKEN_EMBEDDINGS[w]}") for w in TOKEN_INPUT_CODES]),
        ("L2_compound_raw_concepts_18", [(f"rawconcept,{p}", f"{p}\\nb={bias} -> {raw}") for p, (_, _, bias, raw) in PHRASE_RULES.items()]),
        ("L3_category_mappers_3", [
            ("cat,relation", "relation [101,120]\\n-> 7 else 0"),
            ("cat,quantifier", "quantifier [201,220]\\n-> 1 else 0"),
            ("cat,predicate", "predicate [301,320]\\n-> 3 else 0"),
        ]),
        ("L4_figure5_input_7", [
            ("1,1", "C_payload"), ("1,2", "relation_7"), ("1,3", "A1_payload"),
            ("1,4", "quantifier_1"), ("1,5", "A2_payload"), ("1,6", "predicate_3"), ("1,7", "B_payload"),
        ]),
        ("L5_fig5_condition_pass_6", [
            ("2,1", "pass_C_if_relation"), ("2,2", "pass_A1_if_relation"), ("2,3", "pass_A2_if_all"),
            ("2,4", "all_payload"), ("2,5", "have_payload"), ("2,6", "pass_B_if_have"),
        ]),
        ("L6_fig5_compare_prepare_5", [
            ("3,1", "C_payload"), ("3,2", "A2_minus_A1"), ("3,3", "A1_minus_A2"),
            ("3,4", "has_all_have"), ("3,5", "pass_B_if_all"),
        ]),
        ("L7_fig5_validity_check_3", [
            ("4,1", "pass_C_if_A1_eq_A2"), ("4,2", "release_has_if_valid"), ("4,3", "carry_B"),
        ]),
        ("L8_fig5_release_3", [("5,1", "release_C"), ("5,2", "release_has"), ("5,3", "release_B")]),
        ("L9_output_3", [("out,C", "final_C"), ("out,3", "final_has_3"), ("out,B", "final_B")]),
    ]

    fig, ax = plt.subplots(figsize=(36, 19.2))
    ax.set_xlim(-0.7, len(layers) - 0.25)
    ax.set_ylim(-0.7, 22.0)
    ax.axis("off")
    ax.text((len(layers) - 1) / 2, 21.55, "Figure-5 End-to-End Exact Fixed-Parameter Graph", ha="center", fontsize=18, color=text_color)
    ax.text((len(layers) - 1) / 2, 21.05, "token recognition -> compound concept generation -> interval category mapping -> Figure-5 reasoning network", ha="center", fontsize=10.5, color=muted)

    positions = {}
    for li, (lname, nodes) in enumerate(layers):
        ax.text(li, 20.25, lname, ha="center", va="center", fontsize=9.5, color=text_color, fontweight="bold")
        top, bottom = 19.15, 1.35
        ys = [top - i * (top - bottom) / (len(nodes) - 1) if len(nodes) > 1 else (top + bottom) / 2 for i in range(len(nodes))]
        for (nid, label), y in zip(nodes, ys):
            positions[nid] = (li, y)
            ax.add_patch(Circle((li, y), 0.052, fc="white", ec=node_color, lw=1.0, zorder=5))
            bias = ""
            parts = nid.split(",")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                node_key = (int(parts[0]), int(parts[1]))
                if node_key in FIG5_BIAS:
                    bias_value = fmt(FIG5_BIAS[node_key])
                    bias = f"\\nb={bias_value}"
            ax.text(li + 0.035, y, label + bias, ha="left", va="center", fontsize=4.9, color=text_color, zorder=6)

    def width_alpha(w: Fraction | int | float):
        a = abs(float(w))
        if a >= BIG:
            return 1.8, 0.78
        if a >= BIG / 10:
            return 1.6, 0.72
        if a >= 100:
            return 1.25, 0.48
        if a >= 6:
            return 0.95, 0.36
        return 0.55, 0.20

    def draw_edge(src: str, dst: str, w: Fraction | int | float, color=None, label=False):
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        c = color or (pos_color if w >= 0 else neg_color)
        lw, alpha = width_alpha(w)
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.07, y1),
                (x2 - 0.07, y2),
                arrowstyle="-|>",
                mutation_scale=6.5,
                color=c,
                lw=lw,
                alpha=alpha,
                zorder=2,
            )
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, f"w={fmt(F(w))}", ha="center", va="center", fontsize=4.4, color=c, alpha=min(1, alpha + 0.2), zorder=7)

    relation_words = ["belongs", "to", "is", "part", "of", "among", "in", "near", "beside", "before", "after"]
    quantifier_words = ["all", "every", "each", "any", "near", "beside", "before", "after"]
    predicate_words = ["have", "has", "possess", "own", "carry", "contain", "near", "beside", "before", "after"]
    for word in relation_words:
        draw_edge("raw,relation_phrase", f"tok,{word}", 1)
    for word in quantifier_words:
        draw_edge("raw,quantifier_phrase", f"tok,{word}", 1)
    for word in predicate_words:
        draw_edge("raw,predicate_phrase", f"tok,{word}", 1)
    for phrase, (words, weights, _bias, _raw) in PHRASE_RULES.items():
        for word, weight in zip(words, weights):
            draw_edge(f"tok,{word}", f"rawconcept,{phrase}", weight, label=weight != 1)
    for phrase in ["belongs to", "is part of", "is among", "is in"]:
        draw_edge(f"rawconcept,{phrase}", "cat,relation", BIG)
    for phrase in ["all", "every", "each", "any"]:
        draw_edge(f"rawconcept,{phrase}", "cat,quantifier", BIG)
    for phrase in ["have", "has", "possess", "own", "carry", "contain"]:
        draw_edge(f"rawconcept,{phrase}", "cat,predicate", BIG)
    for phrase in ["near", "beside", "before", "after"]:
        draw_edge(f"rawconcept,{phrase}", "cat,relation", -BIG)
        draw_edge(f"rawconcept,{phrase}", "cat,quantifier", -BIG)
        draw_edge(f"rawconcept,{phrase}", "cat,predicate", -BIG)
    for src, dst in [
        ("raw,C", "1,1"), ("cat,relation", "1,2"), ("raw,A1", "1,3"),
        ("cat,quantifier", "1,4"), ("raw,A2", "1,5"), ("cat,predicate", "1,6"), ("raw,B", "1,7"),
    ]:
        draw_edge(src, dst, 1, color=payload_color)
    for src, dst, weight in FIG5_EDGES:
        draw_edge(f"{src[0]},{src[1]}", f"{dst[0]},{dst[1]}", weight, label=abs(weight) != 1)
    for src, dst in [("5,1", "out,C"), ("5,2", "out,3"), ("5,3", "out,B")]:
        draw_edge(src, dst, 1, color=payload_color)

    legend_y = 0.42
    for i, (color, label) in enumerate([(pos_color, "positive weight"), (neg_color, "negative/suppression weight"), (payload_color, "payload/canonical transfer")]):
        x = 2.6 + i * 2.0
        ax.add_patch(FancyArrowPatch((x, legend_y), (x + 0.42, legend_y), arrowstyle="-|>", mutation_scale=8, color=color, lw=1.6, alpha=0.85))
        ax.text(x + 0.50, legend_y, label, va="center", fontsize=7, color=color)
    ax.text((len(layers) - 1) / 2, -0.25, "Node labels show exported fixed parameters. Edge width/opacity increases with |weight|.", ha="center", fontsize=7.2, color=muted)

    fig.savefig(OUT / "parameter_graph.pdf", bbox_inches="tight")
    fig.savefig(OUT / "parameter_graph.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    export_parameters()
    rows, summary = collect_cases()
    export_cases(rows, summary)
    draw_parameter_graph()
    print(f"TOTAL PASSED: {summary['passed']}/{summary['total']}")


if __name__ == "__main__":
    main()
