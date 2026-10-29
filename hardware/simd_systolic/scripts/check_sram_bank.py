#!/usr/bin/env python3
"""Exercise macro-bank arbitration, synchronous reads, masks and reset."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'.build/sram_bank';out.mkdir(exist_ok=True)
inputs=[ROOT/'rtl/Asap7SramMacro.sv',ROOT/'rtl/SramBank.sv',ROOT/'tests/sram_bank_main.cpp']
command=['verilator','--cc','--exe','--build','--assert','-Wno-fatal','-j','4',
         '--top-module','SimdSramBank','-GDEPTH=2048','--Mdir',str(out/'obj'),
         '-o',str(out/'sim'),*map(str,inputs)]
with (out/'build.log').open('w')as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
run=subprocess.run([str(out/'sim')],capture_output=True,text=True,check=True,timeout=60)
report=json.loads(run.stdout)
report['inputs_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs}
report['physical_certified']=False
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
