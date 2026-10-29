# Third-party sources

`vendor/fpnew_{pkg,fma,classifier,rounding}.sv` are copied from CVFPU commit
`77811635cb7e8649f1d9484733ee6a0bdb44a0e6`.
Copyright 2019 ETH Zurich and University of Bologna; authors and per-file
license notices are retained. The files use SHL-0.51 and/or Apache-2.0 as stated
in each header. License texts are included in `vendor/cvfpu_LICENSE.*`.

`vendor/common_cells/{registers,assertions}.svh` are from the CVFPU-pinned
common_cells commit `6aeee85d0a34fedc06c14f04fd6363c9f7b4eeea`; headers and
the upstream license are included. `rtl/lzc.sv` is an independently written
balanced leading-zero tree; it is not the upstream lzc source.

`rtl/SimdFpF2.sv` wraps the upstream combinational FMA.
`scripts/generate_fp.py` and `scripts/generate_fp12.py` derive arithmetic
blocks from the retained FMA. They replace multiplication with an explicitly
pipelined carry-save tree and move metadata, alignment, addition,
normalization and rounding across real register cuts. Generated FMA bodies
retain the upstream SHL-0.51 notice. The added standalone prefix network,
FIFO, Chisel control/datapath and evaluation scripts are original project code.

DRAMSim3 is fetched at commit `29817593b3389f1337235d63cac515024ab8fd6e`.
Its original license remains in the checkout. The local extension is fully
represented by `vendor/patches/dramsim3-drain.patch`: posted write callbacks
keep their upstream meaning; actual read/write bursts and final data drain
are separately observable. No upstream write callback is renamed as a
completed external DRAM transfer.

Nangate45 and tools are downloaded separately into `.deps/`; they are not
redistributed as project source. Their upstream license terms apply. The
Nangate45 files are research reference data and are not a manufacturable PDK.

Ara documentation informed SIMD element splitting and mask handling. This
project does not copy Ara RTL or represent its ALU as a floating-point MAC.
