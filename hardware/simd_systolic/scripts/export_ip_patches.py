#!/usr/bin/env python3
"""Export replayable derivatives against the pinned, unmodified CVFPU FMA."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def export():
    base=ROOT/'vendor/fpnew_fma.sv'
    output=ROOT/'vendor/patches'
    lock=json.loads((ROOT/'dependencies.lock.json').read_text())
    records=[]
    for name,label in [('SimdFpHF.sv','hf8'),('SimdFpHF12.sv','hf12')]:
        target=ROOT/'rtl'/name
        patch=output/f'cvfpu-fma-{label}.patch'
        patch.write_text(''.join(difflib.unified_diff(
            base.read_text().splitlines(keepends=True),
            target.read_text().splitlines(keepends=True),
            fromfile=f'a/rtl/{name}',tofile=f'b/rtl/{name}')))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        records.append(dict(base=str(base.relative_to(ROOT)),target=str(target.relative_to(ROOT)),
            patch=str(patch.relative_to(ROOT)),base_sha256=sha(base),target_sha256=sha(target),patch_sha256=sha(patch)))
    (output/'cvfpu-manifest.json').write_text(json.dumps(dict(
        cvfpu_commit=lock['cvfpu']['commit'],derivatives=records),indent=2)+'\n')


if __name__=='__main__':export()
