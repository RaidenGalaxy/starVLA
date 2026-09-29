#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/starVLA
output=outputs/infinitevl_quant_cpu_rootcause_v1
source_output=outputs/infinitevl_quant_cpu_v2

cd "$repo"
for stage in audit factorial module layer projection; do
    echo "[$(date --iso-8601=seconds)] starting InfiniteVL root-cause stage $stage"
    .venv/bin/python -m experiments.infinitevl_quant_diagnosis.run_cpu_rootcause \
        --stage "$stage" \
        --source-output "$source_output" \
        --output-dir "$output" \
        --resume \
        --local-files-only
done
echo "[$(date --iso-8601=seconds)] InfiniteVL root-cause A-D complete"

