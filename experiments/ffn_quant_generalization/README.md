# Three-model fixed-W4 layerwise generalization

```bash
.venv/bin/python -u -m experiments.ffn_quant_generalization.run_cpu_protocol \
  --stage all --threads 16 --total-hours 240 --round-hours 120 --resume
```

Stages: `audit / rank / sweep / confirm / analyze / report`. One persistent
240-hour deadline includes downloads and preparation; the automatic 120-hour
round boundary never resets it. Twelve hours are reserved for verification.
Concurrent experiment writers are rejected. Reference evidence stays read-only.

The two 36-layer models use both complete 37-point nested curves; Qwen2.5-VL-7B
uses 29 points. Gate/up/down switch together, W4 asymmetric RTN g128, gamma=1,
all other weights reference, CPU FP32 compute on BF16 checkpoint values.
Coverage is `100*N_W4/N_FFN`, 0–100%; BF16 theoretical bit savings is 0.75 times
coverage and excludes metadata. No native low-bit memory/speed claim is made.

ChartQA test and GQA balanced testdev use one question per decoded RGB image.
Historical calibration/evaluation images are excluded. New pilot8/screen32/
confirm128 are image-disjoint, dataset quotas 4/16/64 each; QA32 is a fixed
16-per-dataset confirmation subset. ChartQA human/machine quotas are balanced.
The selection seed is 20261026. Dataset/checkpoint revisions are pinned.
ChartQA training images were used in historical calibration; this is disclosed.

Existing rankings are frozen. 7B is ranked on old diagnostic8 by one-layer
reference restoration on an all-W4 background (28 layers, plus B/W). This old
set is within old screen32 and is not an independent holdout. New test data
never changes sensitivity rankings, masks, random orders, or confirmation points.

Core: 6400 dense-sweep sample pairs, <=864 random-control sample pairs,
3840 independent-confirmation sample pairs. Random controls use 25/50/75%
coverage and PCG64 seeds 42/43/44. Confirmation uses 0/25/50/75%, all-but-one,
and 100%, 10 distinct masks/model. QA uses 5 masks/model, 480 greedy generations,
max 32 new tokens, reusing confirmation teacher losses. The 7B old bridge has
4 masks; the complete old 29-point sweep is optional and needs a full budget
estimate plus 30% headroom before launch.

Inference order: live endpoint/scalar/history audit, old-data 7B ranking, old
bridge, independent confirmation, random controls, full dense curves, QA,
optional 7B old dense sweep, independent analysis/report. No incomplete cell
is promoted to a curve point. A short budget drops optional old scans then QA;
primary NLL sample quotas are never reduced.

NLL is token weighted. Paired dataset-stratified image bootstrap and sign-flip
tests use 10000 replicates. The 36 registered allocation comparisons (3 models,
4 internal coverages, 2 datasets plus pooled) form one Holm family. Curve
points and random seeds are not independent sample units. Local increments
and background interaction are exploratory; QA32 is a directional check.

Results, per-target-token losses, masks, identities, timings, CSV, SVG/PNG,
independent verification and a checksummed sample archive live under
`outputs/ffn_quant_generalization_v1/`. Caches reject changed models, inputs,
code, masks, scorers and backends. Original quantizer/loader modules are reused.

```bash
.venv/bin/python -m pytest -q experiments/ffn_quant_generalization/tests
```

Recovery note (2026-10-28): a CPU FP32 fused cross-entropy sum differed from
historical per-token-then-sum reduction by 1.91e-6 on a 7B bridge sample, while
the selected mask and individual losses were exactly equal. The driver now
uses the historical reduction and checks target positions against the complete
label mask. Original evidence and source are archived under
`recovery/20261028_loss_reduction/`; migration records retain original checksums
and explicitly reaggregate saved token losses under the corrected identity.
The frozen layer order and persistent budget deadline remain unchanged.

Analysis recovery (2026-11-01): persisted JSON encodes layer keys as strings,
whereas runtime intervention fingerprints use integer keys. The analyzer now
reconstructs `Intervention` before checking its fingerprint. All 11,104 core
teacher records matched the original runtime encoding. The version migration
under `recovery/20261101_analysis_identity/` archives all 18,310 records and
preserves every loss, prediction, score, ranking, and the original deadline.
