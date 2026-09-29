InfiniteVL Stage-2 CPU fake-quantization pilot
==============================================

This package runs the pinned ``hustvl/InfiniteVL`` Stage-2 revision through a
pure PyTorch CPU reference backend.  It evaluates quality only.  W2/W3/W4 and
A8 are QDQ simulations and do not provide packed storage or speed claims.

Run the stages in order; every stage is restartable and skips completed sample
artifacts whose configuration and input digests match::

  python -m experiments.infinitevl_quant.run_cpu_protocol \
    --stage preflight --output-dir outputs/infinitevl_quant_cpu_v2 --resume

  python -m experiments.infinitevl_quant.run_cpu_protocol \
    --stage smoke --output-dir outputs/infinitevl_quant_cpu_v2 --resume

  python -m experiments.infinitevl_quant.run_cpu_protocol \
    --stage scope --output-dir outputs/infinitevl_quant_cpu_v2 --resume

  python -m experiments.infinitevl_quant.run_cpu_protocol \
    --stage screen --output-dir outputs/infinitevl_quant_cpu_v2 --resume

  python -m experiments.infinitevl_quant.run_cpu_protocol \
    --stage qig --output-dir outputs/infinitevl_quant_cpu_v2 --resume

  python -m experiments.infinitevl_quant.run_cpu_protocol \
    --stage confirm --output-dir outputs/infinitevl_quant_cpu_v2 --resume

  python -m experiments.infinitevl_quant.run_cpu_protocol \
    --stage report --output-dir outputs/infinitevl_quant_cpu_v2 --resume

The final artifacts are ``INFINITEVL_CPU_QUANT_REPORT_ZH.md``,
``STATE_DRIFT_CURVES.svg`` (or PNG when matplotlib is installed), and
``decision.json``.  Install only the CPU dependencies listed in
``requirements-cpu.txt``; FLA, Triton, flash-attn, and CUDA are deliberately
not required.

``resume_deployment.sh`` is the host-specific recovery supervisor. It verifies
the two pinned LFS SHA256 values and then runs the seven stages in sequence.
A failed stop condition terminates the matrix and writes a partial ``NO_GO``
diagnostic report; it never launches LongSFT or streaming experiments.
