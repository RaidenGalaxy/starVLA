#!/usr/bin/env python3
"""Exhaustively compare actual Chisel narrow expansion to host binary16."""
import fcntl,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    out=ROOT/'.build/expansion_probe';out.mkdir(parents=True,exist_ok=True)
    with(out/'.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with(ROOT/'.build/.elaboration.lock').open('a')as elaborate:
            fcntl.flock(elaborate,fcntl.LOCK_EX)
            subprocess.run(['sbt','-J-Xmx4G','-batch',f'runMain simd.GenerateExpansionProbe {out}'],cwd=ROOT,check=True)
        source=out/'expansion_main.cpp';shutil.copy2(ROOT/'tests/expansion_main.cpp',source)
        rtl=out/'FloatExpansionProbe.sv'
        subprocess.run(['verilator','--cc','--exe','--build','-j','2','-Wno-fatal',
            '--top-module','FloatExpansionProbe','--Mdir',str(out/'obj'),str(rtl),str(source),'-o',str(out/'sim')],check=True)
        report=json.loads(subprocess.check_output([str(out/'sim')],text=True))
        report.update(scope='combinational_chisel_source_expansion',
            rtl_sha256=hashlib.sha256(rtl.read_bytes()).hexdigest(),
            simulator_sha256=hashlib.sha256((out/'sim').read_bytes()).hexdigest(),
            testbench_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            reference='host _Float16 to float for finite inputs; explicit IEEE NaN/infinity payload mapping',
            all_subnormals=True,signaling_nan_payload_preserved=True)
        (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
