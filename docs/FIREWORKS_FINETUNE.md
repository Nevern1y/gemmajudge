# Fireworks Fine-Tune Path

Fireworks is an optional hosted training path for quick iteration. It is not the public core story: GemmaJudge remains Gemma-first and AMD/self-hosted-first, with AMD ROCm/vLLM as the proof path.

## Confirm Support

Fireworks SFT currently uses OpenAI-compatible chat JSONL and supports only models marked `Tunable: true`.

```bash
firectl model get -a fireworks accounts/fireworks/models/gemma-3-12b-it
```

Proceed only if the selected Gemma base model reports `Tunable: true`. If a model has `Supports Lora: true` but not `Tunable: true`, train LoRA elsewhere and import it instead.

## Dataset Format

The canonical GemmaJudge data uses Gemma roles:

```json
{"messages":[{"role":"user","content":"..."},{"role":"model","content":"{...}"}],"metadata":{...}}
```

Fireworks expects OpenAI-style assistant turns, so convert first:

```bash
python scripts/convert_judge_dataset_fireworks.py data/judge_train.jsonl data/fireworks_judge_train.jsonl
python scripts/convert_judge_dataset_fireworks.py data/judge_val.jsonl data/fireworks_judge_val.jsonl
```

The converter intentionally drops `metadata` because Fireworks SFT samples only need `messages`.

## Upload And Launch

```bash
firectl dataset create gemmajudge-judge-train data/fireworks_judge_train.jsonl
firectl dataset create gemmajudge-judge-val data/fireworks_judge_val.jsonl

firectl sftj create \
  --base-model accounts/fireworks/models/gemma-3-12b-it \
  --dataset gemmajudge-judge-train \
  --evaluation-dataset gemmajudge-judge-val \
  --output-model gemmajudge-judge-gemma3-12b-lora \
  --epochs 2.0 \
  --lora-rank 16
```

Monitor:

```bash
firectl sftj get gemmajudge-judge-train
firectl model list
```

Deploy only for private live tests:

```bash
firectl deployment create accounts/<ACCOUNT_ID>/models/gemmajudge-judge-gemma3-12b-lora
```

Delete the dedicated deployment after testing to avoid ongoing cost:

```bash
firectl deployment delete accounts/<ACCOUNT_ID>/deployments/<DEPLOYMENT_ID> --wait
```

## Cost And Cleanup Risk

Fine-tuning and dedicated deployments can consume credits continuously. Check Fireworks pricing before launch, stop/delete deployments after private tests, and never commit API keys or deployment secrets.

For public demo claims, use the committed AMD ROCm proof and recorded artifacts, not Fireworks-hosted training.
