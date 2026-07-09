# GemmaJudge Fine-Tune Evaluation

No tuned-model run has been recorded yet. Do not claim that a tuned GemmaJudge judge
improves judge quality until `report.json` is produced from a real base-vs-tuned run
and the tuned model improves the measured metrics.

This directory should contain only the recorded evaluation proof, not adapters or model
weights. Keep LoRA adapters, merged checkpoints, and caches under `artifacts/`, which is
gitignored.

## AMD ROCm Session Checklist

Run this on the AMD GPU pod before training:

```bash
python scripts/check_rocm_training_env.py --require-gpu
python scripts/build_judge_dataset.py
```

Install the expected training stack if the preflight reports missing packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers peft accelerate vllm openai
```

If the Gemma model is gated, authenticate on the pod with a Hugging Face token from the
runtime environment. Do not commit tokens.

## Train Primary Model

Try Gemma 2 9B first:

```bash
python training/train_gemmajudge_lora.py \
  --model_id google/gemma-2-9b-it \
  --train data/judge_train.jsonl \
  --validation data/judge_val.jsonl \
  --out artifacts/gemmajudge-lora \
  --merge_out artifacts/gemmajudge-merged \
  --max_seq_len 2048 \
  --epochs 2 \
  --learning_rate 2e-5 \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

If 9B does not fit or is too slow, use the smaller proof model:

```bash
python training/train_gemmajudge_lora.py \
  --model_id google/gemma-3-4b-it \
  --train data/judge_train.jsonl \
  --validation data/judge_val.jsonl \
  --out artifacts/gemmajudge-lora \
  --merge_out artifacts/gemmajudge-merged \
  --max_seq_len 2048 \
  --epochs 2 \
  --learning_rate 2e-5 \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

Use QLoRA only after confirming `bitsandbytes-rocm` works on that specific pod image.

## Serve And Evaluate

Use two terminals so the base and tuned models are both available to the comparison
script:

```bash
vllm serve google/gemma-2-9b-it --port 8000 --dtype bfloat16 --max-model-len 2048
vllm serve artifacts/gemmajudge-merged --port 8001 --dtype bfloat16 --max-model-len 2048
```

For a 4B fallback proof, change the base model in the first command and eval command to
`google/gemma-3-4b-it`.

Set dummy keys for local vLLM, then generate the recorded proof artifact:

```bash
export BASE_JUDGE_API_KEY=dummy
export TUNED_JUDGE_API_KEY=dummy

python scripts/eval_judge_model.py \
  --validation data/judge_val.jsonl \
  --base-endpoint http://localhost:8000/v1 \
  --tuned-endpoint http://localhost:8001/v1 \
  --base-model google/gemma-2-9b-it \
  --tuned-model artifacts/gemmajudge-merged \
  --out docs/fine_tune_eval/report.json
```

PowerShell equivalent for the local vLLM keys:

```powershell
$env:BASE_JUDGE_API_KEY="dummy"
$env:TUNED_JUDGE_API_KEY="dummy"
```

## Decision Rule

If the tuned model improves, update this README with the generated summary table and add a
short public proof section in the root README. If it does not improve, keep the public
claim as "fine-tuning-ready judge pipeline" and note that the seed dataset is small and
needs a larger human-reviewed mix.
