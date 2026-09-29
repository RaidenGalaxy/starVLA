#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/starVLA
cache=/home/ubuntu/.cache/huggingface/hub/models--hustvl--InfiniteVL
revision=3d705618aabb1ac60bae75fab6ad0e4679aff58d
snapshot="$cache/snapshots/$revision"
output=outputs/infinitevl_quant_cpu_v2
protocol_started=false

shard1=3ae867c53d50f513b62920b27a772909c6a97df007db223ddf6f312fac926cea
shard2=d761865040fa84262bc34070cc4c5eb98f7d9b1f193c337bb8c0176f25fd41ff
shard1_size=4962256344
shard2_size=3854617496

finalize_shard() {
    local digest=$1
    local expected_size=$2
    local shard_name=$3
    local incomplete="$cache/blobs/$digest.incomplete"
    local complete="$cache/blobs/$digest"
    if [[ ! -f "$complete" ]]; then
        [[ -f "$incomplete" ]] || {
            echo "missing downloaded shard: $incomplete" >&2
            return 1
        }
        local actual_size
        actual_size=$(stat -c '%s' "$incomplete")
        [[ "$actual_size" == "$expected_size" ]] || {
            echo "size mismatch for $shard_name: $actual_size != $expected_size" >&2
            return 1
        }
        echo "$digest  $incomplete" | sha256sum -c -
        mv "$incomplete" "$complete"
    else
        echo "$digest  $complete" | sha256sum -c -
    fi
    if [[ ! -e "$snapshot/$shard_name" ]]; then
        ln -s "../../blobs/$digest" "$snapshot/$shard_name"
    fi
}

echo "[$(date --iso-8601=seconds)] verifying pinned checkpoint"
finalize_shard "$shard1" "$shard1_size" model-00001-of-00002.safetensors
finalize_shard "$shard2" "$shard2_size" model-00002-of-00002.safetensors
echo "[$(date --iso-8601=seconds)] checkpoint SHA256 verification complete"

cd "$repo"
protocol_started=true
report_on_failure() {
    local status=$?
    if [[ "$status" -ne 0 && "$protocol_started" == true ]]; then
        echo "[$(date --iso-8601=seconds)] protocol stopped with status $status; writing NO_GO diagnostics"
        .venv/bin/python -m experiments.infinitevl_quant.run_cpu_protocol \
            --stage report --output-dir "$output" --resume --local-files-only || true
    fi
    exit "$status"
}
trap report_on_failure EXIT

for stage in preflight smoke scope screen qig confirm report; do
    echo "[$(date --iso-8601=seconds)] starting InfiniteVL $stage"
    .venv/bin/python -m experiments.infinitevl_quant.run_cpu_protocol \
        --stage "$stage" --output-dir "$output" --resume --local-files-only
done
trap - EXIT
echo "[$(date --iso-8601=seconds)] InfiniteVL CPU protocol complete"
