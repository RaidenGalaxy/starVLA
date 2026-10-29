# InfiniteVL–Qwen FFN mechanism protocol

This package implements the strong-causal follow-up to
`outputs/ffn_quant_causal_v3`. It treats that directory as immutable input and
writes a new, resumable evidence bundle.

```bash
python -m experiments.ffn_quant_mechanism.run_cpu_protocol \
  --stage audit \
  --parent-output outputs/ffn_quant_causal_v3 \
  --output-dir outputs/ffn_quant_mechanism_v1

python -m experiments.ffn_quant_mechanism.run_cpu_protocol \
  --stage all \
  --parent-output outputs/ffn_quant_causal_v3 \
  --output-dir outputs/ffn_quant_mechanism_v1 \
  --threads 16 --wallclock-hours 120 --resume
```

Stages are `audit`, `calibrate`, `localize`, `transport`, `gdn_mediation`,
`confirm`, and `report`. Formal stages are CPU-only and use W4A16 asymmetric
group-wise fake quantization with group size 128. GDN carry/mediator repair and
error transport are diagnostic counterfactuals, not deployable quantization
methods.
