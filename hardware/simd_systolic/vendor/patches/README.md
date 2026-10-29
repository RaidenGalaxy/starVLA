# IP derivatives

`cvfpu-fma-hf8.patch` and `cvfpu-fma-hf12.patch` describe the complete generated
FMA derivatives against the unmodified `vendor/fpnew_fma.sv` at the commit in
`dependencies.lock.json`. They preserve upstream copyright and license notices.
Apply them in a separate directory after copying that original file to both
target names:

```sh
mkdir -p rtl
cp /path/to/simd_systolic/vendor/fpnew_fma.sv rtl/SimdFpHF.sv
cp /path/to/simd_systolic/vendor/fpnew_fma.sv rtl/SimdFpHF12.sv
patch -p1 < /path/to/simd_systolic/vendor/patches/cvfpu-fma-hf8.patch
patch -p1 < /path/to/simd_systolic/vendor/patches/cvfpu-fma-hf12.patch
```

These files depend on the pinned CVFPU package/classifier/rounding modules and
the project's `PrefixAdd.sv`, `ControlReplicas.sv` and `lzc.sv`. They are separate
SIMD arithmetic cores; they do not replace the public `fpnew_fma` interface.
Their inputs use the project's expanded FP32/FP64 operands and operation encoding.

For implementation changes, edit `scripts/generate_fp.py` and
`scripts/generate_fp12.py`, regenerate both RTL files, then run
`python3 scripts/export_ip_patches.py`. `cvfpu-manifest.json` records the exact
base, target and patch hashes. Timing evidence remains tied to its immutable
input snapshot, rather than to a subsequently regenerated source file.

`dramsim3-drain.patch` applies to the pinned DRAMSim3 checkout and exposes physical
write-data completion/drain observations. `scripts/prepare_dramsim.py` applies
that patch and builds the local shared library.
