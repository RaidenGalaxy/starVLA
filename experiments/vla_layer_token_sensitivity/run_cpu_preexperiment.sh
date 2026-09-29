#!/usr/bin/env bash

# Persistent, restart-safe executor for the preregistered Qwen-OFT CPU study.
# It deliberately serializes checkpoint writers: an existing aria2c process is
# allowed to finish before this script considers restarting the same transfer.

set -Eeuo pipefail

ROOT=/home/ubuntu/starVLA
VENV="$ROOT/.venv/bin/python"
CACHE=/home/ubuntu/.cache/huggingface/hub/models--StarVLA--Qwen-OFT-Bridge-RT-1
BLOB_NAME=51fe8d22c8d57116c2f59c5fdb24323fa3411149e888b807edba99b8354e0861
INCOMPLETE="$CACHE/blobs/$BLOB_NAME.incomplete"
CONTROL="$INCOMPLETE.aria2"
FINAL_BLOB="$CACHE/blobs/$BLOB_NAME"
EXPECTED_SIZE=8215912766
EXPECTED_SHA256=51fe8d22c8d57116c2f59c5fdb24323fa3411149e888b807edba99b8354e0861
URL=https://huggingface.co/StarVLA/Qwen-OFT-Bridge-RT-1/resolve/main/checkpoints/steps_10000_pytorch_model.pt
CONFIG="$CACHE/snapshots/fd4fffb40684c2b1b9d8f66672b3ba691bc137a1/config.yaml"
BASE_VLM="$ROOT/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
OUT="$ROOT/outputs/vla_layer_token_sensitivity"
MANIFEST="$OUT/bridge_subset/manifest.jsonl"
SCAN="$OUT/qwen_oft_anchor_scan.jsonl"
LOG="$OUT/cpu_preexperiment.log"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

quarantine() {
  local path=$1
  if [[ -e "$path" ]]; then
    local stamp
    stamp=$(date -u +'%Y%m%dT%H%M%SZ')
    mv -- "$path" "$path.quarantine.$stamp"
    log "quarantined $path"
  fi
}

verify_file() {
  local path=$1
  [[ -f "$path" ]] || return 1
  local size digest
  size=$(stat -c '%s' "$path")
  [[ "$size" == "$EXPECTED_SIZE" ]] || return 1
  digest=$(sha256sum "$path" | awk '{print $1}')
  [[ "$digest" == "$EXPECTED_SHA256" ]]
}

run_aria2() {
  aria2c \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=10M \
    --file-allocation=none \
    --max-tries=0 \
    --retry-wait=5 \
    --timeout=60 \
    --connect-timeout=30 \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --summary-interval=20 \
    --dir="$CACHE/blobs" \
    --out="$BLOB_NAME.incomplete" \
    "$URL"
}

finish_checkpoint() {
  if [[ -f "$FINAL_BLOB" ]]; then
    log "validating existing final blob"
    if verify_file "$FINAL_BLOB"; then
      return
    fi
    quarantine "$FINAL_BLOB"
  fi

  while pgrep -x aria2c >/dev/null; do
    log "waiting for the existing aria2c writer"
    sleep 30
  done

  while [[ -e "$CONTROL" || ! -f "$INCOMPLETE" ]]; do
    log "starting or resuming aria2 checkpoint transfer"
    if ! run_aria2; then
      log "aria2 exited abnormally; retaining control state and retrying"
      sleep 5
    fi
    if [[ ! -e "$CONTROL" && -f "$INCOMPLETE" ]]; then
      break
    fi
  done

  log "checking checkpoint size and SHA256"
  if ! verify_file "$INCOMPLETE"; then
    log "checkpoint validation failed; quarantining partial state"
    quarantine "$INCOMPLETE"
    quarantine "$CONTROL"
    run_aria2
    if [[ -e "$CONTROL" ]] || ! verify_file "$INCOMPLETE"; then
      log "fresh checkpoint transfer did not validate"
      return 1
    fi
  fi
  mv -- "$INCOMPLETE" "$FINAL_BLOB"
  log "checkpoint atomically promoted to final blob"
}

register_snapshot() {
  HF_HUB_DISABLE_XET=1 "$VENV" - <<'PY'
import time
from huggingface_hub import hf_hub_download

for attempt in range(1, 51):
    try:
        path = hf_hub_download(
            repo_id="StarVLA/Qwen-OFT-Bridge-RT-1",
            filename="checkpoints/steps_10000_pytorch_model.pt",
        )
        print(path)
        break
    except KeyboardInterrupt:
        raise
    except Exception:
        if attempt == 50:
            raise
        time.sleep(5)
PY
}

write_integrity_record() {
  local snapshot_path=$1
  "$VENV" - "$FINAL_BLOB" "$snapshot_path" "$OUT/checkpoint_integrity.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

blob, snapshot, output = map(Path, sys.argv[1:])
digest = hashlib.sha256()
with blob.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
payload = {
    "status": "verified",
    "size": blob.stat().st_size,
    "sha256": digest.hexdigest(),
    "blob_path": str(blob.resolve()),
    # Keep the snapshot-facing path for auditability rather than resolving the
    # symlink back to the content-addressed blob recorded separately above.
    "snapshot_path": str(snapshot.absolute()),
    "verified_at": datetime.now(timezone.utc).isoformat(),
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
PY
}

validate_smoke() {
  "$VENV" - "$OUT/qwen_oft_smoke.json" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["status"] == "complete"
assert payload["layers_total"] == 36
assert payload["action_shape"] == [1, 16, 7]
assert payload["token_counts"]["action_query"] == 16
assert all(payload["token_counts"][name] > 0 for name in ("vision", "language", "action_query"))
assert len(payload["rows"]) == 2
assert {(row["layer"], row["operation"]) for row in payload["rows"]} == {(0, "skip"), (0, "w8")}

def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True

assert finite(payload)
print("smoke validation passed")
PY
}

validate_manifest() {
  "$VENV" - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 32
assert len({row["sample_id"] for row in rows}) == 32
counts = Counter(int(row["episode_index"]) for row in rows)
assert len(counts) == 8 and set(counts.values()) == {4}
assert all(Path(row["image_path"]).is_file() for row in rows)
assert all(len(row["action"]) == 7 and len(row["state"]) == 7 for row in rows)
assert all(str(row["instruction"]).strip() for row in rows)
assert max(float(row["video_timestamp_error"]) for row in rows) <= 0.25
print(json.dumps({"manifest_rows": 32, "episodes": 8, "instructions": len({row['instruction'] for row in rows})}))
PY
}

validate_scan() {
  local expected_samples=$1
  "$VENV" - "$SCAN" "$expected_samples" <<'PY'
import json
import math
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
expected_samples = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
keys = {(row["sample_id"], row["base_state"], int(row["layer"]), row["operation"]) for row in rows}
assert len(keys) == len(rows)

def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True

assert finite(rows)
sample_ids = {row["sample_id"] for row in rows}
if expected_samples == 32:
    assert len(rows) == 512 and len(sample_ids) == 32
else:
    assert len(rows) >= 64 and len(sample_ids) >= 4
assert any(
    row["base_state"] == "fp" and row["operation"] == "skip"
    and row["metrics"]["vs_fp"]["action_mse"] > 0
    for row in rows
)
w4_refs = [row for row in rows if row["base_state"] == "w4" and row["operation"] == "reference"]
assert w4_refs and any(row["metrics"]["vs_fp"]["action_mse"] > 0 for row in w4_refs)
assert all(row["quantization"]["linears"] == 252 for row in w4_refs)
reference_dir = path.parent / f"{path.stem}_references"
assert len(list(reference_dir.glob("*.npz"))) >= expected_samples
episodes = Counter(int(row["episode_index"]) for row in rows if row["base_state"] == "fp" and row["operation"] == "reference")
assert len(episodes) >= min(expected_samples, 8)
print(json.dumps({"rows": len(rows), "samples": len(sample_ids), "reference_npz": len(list(reference_dir.glob('*.npz')))}))
PY
}

main() {
  cd "$ROOT"
  log "persistent CPU pre-experiment executor started"
  finish_checkpoint
  local checkpoint
  checkpoint=$(register_snapshot | tail -1)
  [[ -f "$checkpoint" ]]
  write_integrity_record "$checkpoint"
  log "checkpoint registered at $checkpoint"

  log "running strict single-image smoke"
  "$VENV" -m experiments.vla_layer_token_sensitivity.qwen_oft_probe \
    --checkpoint "$checkpoint" \
    --config "$CONFIG" \
    --base-vlm "$BASE_VLM" \
    --image "$ROOT/assets/starvla_simpleEnv.png" \
    --instruction "pick up the object" \
    --layers 0 \
    --ops skip,w8 \
    --threads 16 \
    --output "$OUT/qwen_oft_smoke.json"
  validate_smoke

  log "downloading and extracting the 32-sample Bridge subset"
  HF_HUB_DISABLE_XET=1 "$VENV" -m experiments.vla_layer_token_sensitivity.bridge_subset \
    --output-dir "$OUT/bridge_subset" \
    --episodes 8 \
    --fractions 0.2,0.4,0.6,0.8 \
    --seed 42
  validate_manifest

  log "running the four-episode pilot"
  "$VENV" -m experiments.vla_layer_token_sensitivity.qwen_oft_manifest_probe \
    --checkpoint "$checkpoint" \
    --config "$CONFIG" \
    --base-vlm "$BASE_VLM" \
    --manifest "$MANIFEST" \
    --output "$SCAN" \
    --layers 0,6,12,18,24,30,34 \
    --base-states fp,w4 \
    --group-size 128 \
    --threads 16 \
    --limit 4
  validate_scan 4

  log "pilot passed; resuming the complete 32-sample anchor scan"
  "$VENV" -m experiments.vla_layer_token_sensitivity.qwen_oft_manifest_probe \
    --checkpoint "$checkpoint" \
    --config "$CONFIG" \
    --base-vlm "$BASE_VLM" \
    --manifest "$MANIFEST" \
    --output "$SCAN" \
    --layers 0,6,12,18,24,30,34 \
    --base-states fp,w4 \
    --group-size 128 \
    --threads 16
  validate_scan 32

  log "running preregistered episode-bootstrap gate analysis"
  "$VENV" -m experiments.vla_layer_token_sensitivity.analyze_qwen_oft_scan \
    --input "$SCAN" \
    --output-json "$OUT/qwen_oft_gate_result.json" \
    --output-report "$OUT/QWEN_OFT_GATE_REPORT_ZH.md" \
    --bootstrap-samples 2000 \
    --seed 42

  "$VENV" -m pytest -q experiments/vla_layer_token_sensitivity/tests
  log "all stages complete"
}

trap 'log "FAILED at line $LINENO with exit code $?"' ERR
main "$@"
