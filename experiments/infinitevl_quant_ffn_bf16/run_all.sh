#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/starVLA
output=outputs/infinitevl_quant_ffn_bf16_v1
source_output=outputs/infinitevl_quant_cpu_rootcause_v1

cd "$repo"
for stage in audit smoke screen confirm report; do
    echo "[$(date --iso-8601=seconds)] starting InfiniteVL FFN-reference stage $stage"
    .venv/bin/python -m experiments.infinitevl_quant_ffn_bf16.run_cpu_protocol \
        --stage "$stage" \
        --source-rootcause-output "$source_output" \
        --output-dir "$output" \
        --resume \
        --local-files-only
done
echo "[$(date --iso-8601=seconds)] InfiniteVL FFN-reference protocol complete"

