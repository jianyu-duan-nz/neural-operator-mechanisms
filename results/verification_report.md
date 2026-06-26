Figure-5 End-to-End Exact Verification
======================================

Pipeline:
  raw word/phrase inputs
  -> token recognition
  -> compound concept generation
  -> interval/category normalization
  -> Figure-5 reasoning core
  -> final output

Result:
  TOTAL PASSED: 16680/16680

Breakdown:
- exhaustive_phrase_combination: 11664/11664
- predicate_interval: 999/999
- quantifier_interval: 999/999
- random_positive_integer: 2000/2000
- raw_phrase_generation: 18/18
- relation_interval: 999/999
- token_code_embedding_uniqueness: 1/1

Expected behavior:
  Valid cases output [C, 3, B].
  Invalid cases output [0, 0, 0].

The complete parameter specification is exported in parameters.json and the
CSV files in this results directory.
