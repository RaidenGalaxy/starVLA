# Fixed-W4 FFN compression / quality curves

Run from the repository root:

```bash
.venv/bin/python -m experiments.ffn_quant_compression.run_cpu_protocol \
  --stage all --threads 16 --wallclock-hours 120 --reserve-hours 5 --resume \
  --output-dir outputs/ffn_quant_compression_v1
```

The independent stages are `audit / sweep / analyze / report`. One persistent
absolute deadline covers the whole run; resume never resets it. Model calls
are checked at cooperative boundaries, with five hours reserved for analysis.
Concurrent writers are rejected. Historical code and results are read-only.

The 36-layer sensitivity ranking is frozen from the preceding diagnostic8.
At q W4 layers, sensitive-first quantizes the first q entries; sensitive-protected
quantizes the last q entries. Gate/up/down are switched together. All other
weights retain checkpoint values. W4 asymmetric RTN g128 is unchanged; gamma=1.

For each model, 72 distinct masks cover the two 37-point NLL curves on screen32.
Three NumPy permutations (42,43,44) add 12 distinct masks at q=0,3,18,33,35,36.
The two QA curves use these six q values on confirm192, ten distinct masks per
model. Endpoints are shared. QA masks run on BOTH models before NLL scans.
Random controls may be explicitly deferred to preserve the analysis reserve.
Incomplete conditions are never promoted to curve points or statistical tests.

Compression is theoretical FFN weight-bit savings relative to BF16:
`1-(16*N_reference+4*N_w4)/(16*N_ffn) = q/48`. Quantization metadata is excluded.
This is CPU FP32 fake quantization, not native packed low-bit memory or speed.
The QA cohort reuses the historical confirm192 sample identities; it is not a
new external holdout. The frozen scorer is replayed during final verification.

Imports require matching numerical source, model, input, runtime, scorer and
data identities, plus live teacher/generation sentinels. Imported rows retain
source hashes and provenance, and new baseline-reference files bind the current
audit identity. Identity mismatches disable historical imports; corrupt evidence
raises an error. All curves are keyed by their complete 36-bit mask and input.

QA figures use sample-bootstrap pointwise 95% intervals. NLL random bands show
the three-permutation range, not confidence intervals. Spearman correlations
are descriptive paired-bootstrap estimates, with average tie ranks rounded at
12 decimal places to suppress FP64 aggregate roundoff. Constant curves are
undefined. The 16 confirm192 allocation contrasts (two models, four interior
q values, two metrics) use two-sided sample sign-flips and one Holm family.

Deliverables are `report/qa_vs_compression.svg/png`,
`report/nll_vs_compression.svg/png`, `report/curve_points.csv`,
`report/report_zh.md`, raw sample records and `verification.json`.

Tests:

```bash
.venv/bin/python -m pytest -q experiments/ffn_quant_compression/tests
```
