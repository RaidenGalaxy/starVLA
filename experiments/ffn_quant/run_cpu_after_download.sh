#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

model_dir="playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
smoke_dir="outputs/ffn_quant/qwen2_5_vl_3b_smoke"
cpu_dir="outputs/ffn_quant/qwen2_5_vl_3b"

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  .venv/bin/python -m experiments.ffn_quant.run_ablation_safe \
    --model-path "$model_dir" \
    --preset smoke \
    --resume \
    --output-dir "$smoke_dir"

.venv/bin/python -c "import json; from pathlib import Path; root=Path('$smoke_dir'); inv=json.loads((root/'module_inventory.json').read_text()); metrics=json.loads((root/'metrics.json').read_text()); assert inv['linear_modules']==108; assert inv['target_parameters']==2434793472; rows=metrics['awq_w4_last1']['awq_search']; assert rows and all(row['awq_normalized_mse'] <= row['rtn_normalized_mse'] + 1e-12 for row in rows); print('smoke acceptance: PASS')"

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  .venv/bin/python -m experiments.ffn_quant.run_ablation_safe \
    --model-path "$model_dir" \
    --preset cpu \
    --resume \
    --output-dir "$cpu_dir"
