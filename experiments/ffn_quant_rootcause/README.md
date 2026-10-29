# InfiniteVL FFN root-cause protocol

Run from the repository root using the existing CPU environment:

```bash
.venv/bin/python -m pip install -r experiments/ffn_quant_rootcause/requirements.txt
.venv/bin/python -m experiments.ffn_quant_rootcause.run_cpu_protocol \
  --stage all --threads 16 --wallclock-hours 120 --resume \
  --output-dir outputs/ffn_quant_rootcause_v1
```

Stages: `audit / ratio / local / dose / propagation / confirm / report`.
Run one stage with `--stage NAME --resume`; its prerequisites must be complete.
`--stage report` regenerates a partial report without running model inference.
The CLI rejects concurrent writers and incompatible resumed identities.

Historical packages and outputs are read-only. Models, all input images, code,
runtime versions, scorer annotations and historical evidence are content-addressed.
Every new sample row binds its exact intervention and processed-input digest;
stage summaries and sample rows have checksums. New B/W baselines are computed
under the current identity; historical metric rows are never silently imported.

Round 1 (`audit` through `dose`) and round 2 (`propagation` and `confirm`) each
have a persistent absolute wall-clock deadline, capped at 120 hours. Resume does
not reset a deadline. Enforcement is cooperative before model calls and after
committed work; an in-flight call is not hard-interrupted. An exhausted round
leaves resumable evidence and a partial report, without a completion marker.

The calibration is qig32; localization uses diagnostic8; screening uses screen32;
confirmation uses the historical confirm192 split. Split disjointness also checks
image/question hashes. The top three layers are selected on diagnostic8. At most
two confirm interventions are selected by InfiniteVL screen32 token-weighted
absolute NLL recovery, locked before any confirmation model input is processed, with corresponding
Qwen interventions and matched controls where applicable. GDN mediation is run
unconditionally and never uses local GDN/SWA gain as a gate. Oracle repair is
teacher-forced only, with reference self-repair controls.

Ratios are FP64 diagnostic reductions of FP32 compute. Complete sequences and
at-most-256-token historical traces are labeled separately. Visual, text-prompt,
answer-prediction and all-valid masks exclude padding; groups may overlap at the
last prompt token. Zero denominators are null, not epsilon-clamped. Full sample
distributions, numerators, denominators and achieved transport errors are retained.

Outputs: `status.json`, per-stage `summary.json`, bound sample rows,
`confirm/locked_interventions.json`, and `report/report_zh.md` with CSV/SVG ratio
artifacts. Reports do not promote partial matrices or isolated significant
effects to a final root-cause claim. All results are CPU FP32 QDQ quality
simulations, not native low-bit performance claims.

Matrices finish one complete condition at a time. Observed cell runtime and a
remaining-work estimate are saved in each stage's schedule files. Required dose
and bidirectional transport conditions run before optional direction/channel
families. A whole optional family (all three random seeds together) can be
deferred if its forecast would consume the reserve for the other model's required
matrix. Such omissions are explicit in the stage summary/report and cannot be
used to support a mechanism claim. Interrupted cells are never aggregated.

Tests:

```bash
.venv/bin/python -m pytest -q experiments/ffn_quant_rootcause/tests
```
