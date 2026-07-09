# GemmaJudge Judge Dataset Report

Generated from committed GemmaJudge artifacts plus a small Apache-2.0 curated seed set.

## Summary

- Total examples: 156
- Train examples: 100
- Validation examples: 56
- Output schema: `gemmajudge-judge-v1`
- Deduplication key: sha256(failure_mode + user_prompt + target_response)
- Conservative fallback labels retained: 1
- Score-3 ambiguous examples forced to validation/eval: 15

## By Source

- amd_proof_w7900: 5
- gemmajudge_curated_seed: 111
- real_runs_leaderboard: 40

## By Failure Mode

- bias: 37
- hallucination: 82
- jailbreak: 37

## By Label

- non_violation: 99
- violation: 57

## Split Balance

- train: bias=25, hallucination=51, jailbreak=24
- validation: bias=12, hallucination=31, jailbreak=13

## Licenses

- `amd_proof_w7900`: Apache-2.0 project artifact.
- `real_runs_leaderboard`: Apache-2.0 project artifact.
- `gemmajudge_curated_seed`: Apache-2.0 hand-curated/synthetic seed examples.

No external benchmark rows are included in this seed build. TruthfulQA and BBQ remain
reserved for held-out evaluation unless a future import script explicitly marks them as
non-training data.

## Limitations

- Recorded artifact labels are judge-generated, not independently human-labeled.
- The dataset is fine-tuning-ready but intentionally small; use external imports for a
  production-quality 300-1000+ example mix.
- Validation split is group-aware by target model or curated family where possible, but
  some attack prompt patterns repeat across different target responses.
