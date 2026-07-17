#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

model_dir="playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
smoke_dir="outputs/ffn_quant/qwen2_5_vl_3b_smoke"
cpu_dir="outputs/ffn_quant/qwen2_5_vl_3b"

shard1_hash="d8cd65a1c446dc687c9b7f80caf1a2ce5f859f301e17c25cb73347def030f639"
shard2_hash="1a46f4b515d44f2f64afef11a2999df3ed3317083572a17b3a5256f35ca68ccf"
shard1_sha="41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8"
shard2_sha="365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4"

mkdir -p "$model_dir" "$smoke_dir" "$cpu_dir"

if ! printf '%s  %s\n%s  %s\n' \
    "$shard1_sha" "$model_dir/model-00001-of-00002.safetensors" \
    "$shard2_sha" "$model_dir/model-00002-of-00002.safetensors" | sha256sum --check --status; then
  aria2c \
    --continue=true \
    --max-concurrent-downloads=2 \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=4M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --summary-interval=60 \
    --dir="$model_dir" \
    -Z \
    'https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00001-of-00002.safetensors?download=true' \
    'https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/model-00002-of-00002.safetensors?download=true'
  mv "$model_dir/$shard1_hash" "$model_dir/model-00001-of-00002.safetensors"
  mv "$model_dir/$shard2_hash" "$model_dir/model-00002-of-00002.safetensors"
fi

printf '%s  %s\n%s  %s\n' \
  "$shard1_sha" "$model_dir/model-00001-of-00002.safetensors" \
  "$shard2_sha" "$model_dir/model-00002-of-00002.safetensors" | sha256sum --check

HF_HUB_DISABLE_XET=1 .venv/bin/hf download \
  Qwen/Qwen2.5-VL-3B-Instruct tokenizer.json \
  --local-dir "$model_dir"

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  .venv/bin/python -m experiments.ffn_quant.run_ablation \
    --model-path "$model_dir" \
    --preset smoke \
    --resume \
    --output-dir "$smoke_dir"

.venv/bin/python -c "import json; from pathlib import Path; root=Path('$smoke_dir'); inv=json.loads((root/'module_inventory.json').read_text()); metrics=json.loads((root/'metrics.json').read_text()); assert inv['linear_modules']==108; assert inv['target_parameters']==2434793472; rows=metrics['awq_w4_last1']['awq_search']; assert rows and all(row['awq_normalized_mse'] <= row['rtn_normalized_mse'] + 1e-12 for row in rows); print('smoke acceptance: PASS')"

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  .venv/bin/python -m experiments.ffn_quant.run_ablation \
    --model-path "$model_dir" \
    --preset cpu \
    --resume \
    --output-dir "$cpu_dir"

