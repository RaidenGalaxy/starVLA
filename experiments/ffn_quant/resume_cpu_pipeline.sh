#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

model_dir="playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
shard1="$model_dir/model-00001-of-00002.safetensors"
shard2="$model_dir/model-00002-of-00002.safetensors"

while [[ ! -f "$shard1" || ! -f "$shard2" ]]; do
  bash experiments/ffn_quant/run_cpu_pipeline.sh
  status=$?
  if [[ $status -eq 0 ]]; then
    exit 0
  fi
  if [[ -f "$shard1" && -f "$shard2" ]]; then
    echo "Pipeline failed after model download; not retrying non-download stages." >&2
    exit "$status"
  fi
  echo "Download interrupted (status $status); retrying from aria2 control files in 30 seconds." >&2
  sleep 30
done

exec bash experiments/ffn_quant/run_cpu_pipeline.sh
