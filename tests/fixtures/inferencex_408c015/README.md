# InferenceX native recipe fixtures

Selected source files copied unmodified from SemiAnalysisAI/InferenceX commit
`408c015be4b22d14c69518643609669405507077` (Apache-2.0; see LICENSE).
The master config is reduced to four original entries and runners to its hardware mapping.
Tests invoke the real pure recipe validator and golden acceptance functions.
Client runtime files are generated as explicit sentinels by the fixture factory;
these CPU tests do not execute a server or claim canonical GPU replay coverage.

`benchmarks/benchmark_lib_eval.sh` contains the unmodified `check_env_vars`,
`run_lm_eval`, and `run_eval` functions from the same commit's `benchmark_lib.sh`.
The generic eval boundary test executes these functions with only the actual
model-client invocation replaced, so upstream workflow-variable requirements
are checked without starting a GPU server or installing eval dependencies.
