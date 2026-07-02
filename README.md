# Neural Operator Mechanisms in AI Reasoning

Code, parameters, and exact-verification artifacts accompanying the paper:

**The Reasoning Mechanism of AI Neural Networks**
*A Clear, Neuronal-Level Account of Language, Reasoning, and Other Semantic Transformations*
Jianyu Duan and Mingjun Duan — AEQ AI Research Institute, New Zealand

📄 Paper: [`The_Reasoning_Mechanism_of_AI_Neural_Networks.pdf`](https://jianyu-duan-nz.github.io/neural-operator-mechanisms/The_Reasoning_Mechanism_of_AI_Neural_Networks.pdf)

📊 Experimental analysis report: [`Experimental_Analysis_Report.pdf`](https://jianyu-duan-nz.github.io/neural-operator-mechanisms/Experimental_Analysis_Report.pdf)

📰 News webpage: [English-edition news page](https://jianyu-duan-nz.github.io/neural-operator-mechanisms/)

> **Scope of this repository.** The paper rests on three layers of evidence — an explicit
> parameter-level construction, free-trained dense networks and a Transformer, and a
> reading of large-model phenomena. **This repo contains only the first layer:** the
> self-contained exact-construction-and-verification experiment for the syllogism circuit
> (the 16,680-test pipeline). The trained-network experiments are summarized in
> [How this fits the paper](#how-this-fits-the-paper) and reported in full in a companion
> experiment report referenced in the paper; their code and artifacts are not included here.

---

## What this is

How a neural network actually *reasons* — not what it outputs, but the computation
that constitutes an act of reasoning — has remained a black box. This paper argues
that language understanding and reasoning are **definite computations over internal
semantic structure**, carried out by **cognitive neural operators**: small groups of
neurons (weights, biases, ReLU-equivalent activations, and connection paths) that
recognize concepts, relations, and roles, compare them, suppress invalid cases, and
transform valid semantic structures into new ones.

The central claim is made concrete with a **parameter-level circuit** that realizes
the categorical syllogism

```
[ C, belongs-to, A ] + [ all, A, have, B ]  -->  [ C, has, B ]
```

The complete word-and-phrase-to-conclusion pipeline is specified down to individual
weight parameters and **passes all 16,680 exact simulation tests**. This repository
contains that construction and its verification.

The experiment in this repository is the **exact-construction-and-verification** of that
circuit — not a training run: the weights, biases, token codes, concept-generation rules,
category mappers, and reasoning-core connections are specified explicitly and then tested
exhaustively over a controlled set of valid and invalid inputs. The circuit is a concrete
*demonstration* of the operator account the paper derives — the operators come first, and
this circuit instantiates and verifies them — rather than a single example the theory is
extrapolated from. The trained-network experiments that independently corroborate the same
operators are summarized below but live elsewhere (see the scope note above).

## The verified pipeline

```
raw word/phrase inputs
  -> token recognition
  -> compound concept generation
  -> interval/category normalization
  -> Figure-5 reasoning core
  -> final output
```

- Valid premise pairs produce `[C, 3, B]` (where `3` is the internal predicate value
  for `have`/`has`).
- Every violation of the relation, quantifier, predicate, or middle-term match is
  suppressed to `[0, 0, 0]`.

So the conclusion is **checked, not asserted**: it is released only when the relation
(`belongs-to`), the quantifier (`all`), the predicate (`have`), and the equality of
the shared middle term all hold.

## Repository contents

| Path | Description |
| --- | --- |
| [`run_fig5_end_to_end_exact.py`](run_fig5_end_to_end_exact.py) | Self-contained script that builds the explicit circuit and runs the full exact verification. |
| [`results/`](results/) | Generated parameter specification and verification artifacts (see below). |
| [`The_Reasoning_Mechanism_of_AI_Neural_Networks.pdf`](https://jianyu-duan-nz.github.io/neural-operator-mechanisms/The_Reasoning_Mechanism_of_AI_Neural_Networks.pdf) | The full paper. |
| [`Experimental_Analysis_Report.pdf`](https://jianyu-duan-nz.github.io/neural-operator-mechanisms/Experimental_Analysis_Report.pdf) | Companion experimental analysis report (controlled experiments testing the operator account). |
| [English-edition news page](https://jianyu-duan-nz.github.io/neural-operator-mechanisms/) | Companion English-edition write-up of the verification (rendered via GitHub Pages). |
| [`requirements.txt`](requirements.txt) | Python dependencies (only `matplotlib`, for the exported graph). |

## Run

Requires Python 3 (standard library plus `matplotlib`).

```bash
pip install -r requirements.txt
python3 run_fig5_end_to_end_exact.py
```

The script writes all outputs to `results/`.

## Generated outputs

- `results/parameters.json` — complete parameter specification.
- `results/token_parameters.csv` — token input codes and recognized embeddings.
- `results/phrase_rules.csv` — compound-concept generation rules.
- `results/category_mappers.csv` — interval-to-category mappings.
- `results/figure5_biases.csv` — Figure-5 node biases.
- `results/figure5_edges.csv` — Figure-5 weighted edges.
- `results/verification_cases.csv` — all 16,680 verification checks.
- `results/verification_summary.json` — pass/fail summary.
- `results/verification_report.md` — short human-readable report.
- `results/run_log.txt` — compact run log.
- `results/parameter_graph.png` / `results/parameter_graph.pdf` — exported parameter graph.

The current artifacts record **16,680 / 16,680** tests passing, broken down as:

| Test family | Passed |
| --- | --- |
| Exhaustive phrase combinations | 11,664 / 11,664 |
| Predicate interval | 999 / 999 |
| Quantifier interval | 999 / 999 |
| Relation interval | 999 / 999 |
| Random positive integers | 2,000 / 2,000 |
| Raw phrase generation | 18 / 18 |
| Token-code embedding uniqueness | 1 / 1 |

## How this fits the paper

The paper develops the operator account in three layers of evidence:

1. **Explicit construction (this repo).** A parameter-level circuit, verified exactly
   end to end over all valid and invalid cases.
2. **Free-trained networks.** Dense networks trained on the final conclusion alone, and
   a standard attention-plus-ReLU Transformer, are shown to realize the same operator
   functions — as clean modules or as *distributed / split / cross-shared / compressed /
   stretched* topological variants of the same operators.
3. **Large-model phenomena.** The same neural-operator language gives a concrete reading
   of emergence, scaling, generalization, and hallucination, and reframes feature- and
   circuit-level interpretability as a map of underlying operators.

The trained-network evidence (paper §10) shows the same operators arising without being
hand-built:

- **Free training.** Dense networks trained on final outputs alone form the recognition,
  gating, and mismatch-blocking units on their own, and the same operator topology
  reappears in a standard attention-plus-ReLU Transformer (confirmed by causal
  interventions).
- **Robustness.** The operators are learned equivalently under ReLU, GELU, SiLU, SwiGLU,
  and attention-style edges, and across a range of reasoning types beyond the categorical
  syllogism (propositional, predicate, nested, fuzzy, modal, and analogical).
- **Discriminating tests.** Structure-reusing networks generalize to unseen chain lengths
  where unstructured networks of comparable size do not, and only the genuinely required
  conditions suppress the output — predictions that could have failed but did not.

Full details of these trained-network experiments are reported in a companion experiment
report referenced in the paper.

## Citation

```bibtex
@article{duan2026reasoning,
  title   = {The Reasoning Mechanism of AI Neural Networks:
             A Clear, Neuronal-Level Account of Language, Reasoning,
             and Other Semantic Transformations},
  author  = {Duan, Jianyu and Duan, Mingjun},
  institution = {AEQ AI Research Institute, New Zealand},
  year    = {2026}
}
```

## Contact

Jianyu Duan — `jyduan.aeq@gmail.com`
