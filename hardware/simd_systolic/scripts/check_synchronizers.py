#!/usr/bin/env python3
"""Verify retained two-stage reset/toggle structures and asynchronous behavior."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check():
    out=ROOT/'.build/synchronizers';out.mkdir(exist_ok=True)
    rtl=ROOT/'rtl/ControlReplicas.sv';structures=[]
    for top,reset,initial in [('SimdResetSynchronizer','externalReset',1),('SimdToggleSynchronizer','reset',0)]:
        path=out/(top+'.json')
        cmd=f'read_verilog -sv {rtl}; hierarchy -top {top}; proc; opt; write_json {path}'
        result=subprocess.run(['yosys','-Q','-T','-p',cmd],capture_output=True,text=True,check=True)
        (out/(top+'.log')).write_text(result.stdout+result.stderr)
        module=json.loads(path.read_text())['modules'][top];nets=module['netnames'];cells=module['cells']
        stages=[]
        for name in ('first','second'):
            bits=nets[name]['bits']
            found=[c for c in cells.values()if c['connections'].get('Q')==bits]
            assert len(found)==1 and nets[name]['attributes'].get('ASYNC_REG')=='TRUE'
            cell=found[0];stages.append(cell)
            assert cell['type']=='$adff'and cell['connections']['CLK']==nets['clock']['bits']
            assert cell['connections']['ARST']==nets[reset]['bits']
            assert int(cell['parameters']['ARST_VALUE'],2)==initial
        assert stages[0]['connections']['D']==(['0']if top=='SimdResetSynchronizer'else nets['data']['bits'])
        assert stages[1]['connections']['D']==nets['first']['bits']and nets['value']['bits']==nets['second']['bits']
        for cell in cells.values():
            for port,bits in cell['connections'].items():
                if cell['port_directions'][port]=='input'and set(bits)&set(nets['first']['bits']):
                    assert cell is stages[1]and port=='D','first stage drives functional logic'
        structures.append(dict(module=top,stages=2,attributes_preserved=True,first_stage_only_drives_second=True))
    with(out/'build.log').open('w')as log:
        subprocess.run(['verilator','--cc','--exe','--build','--assert','-j','4','--top-module','SynchronizerProbe',
            '--Mdir',str(out/'obj'),str(rtl),str(ROOT/'tests/SynchronizerProbe.sv'),
            str(ROOT/'tests/synchronizers_main.cpp'),'-o',str(out/'sim')],stdout=log,stderr=subprocess.STDOUT,check=True)
    dynamic=subprocess.run([str(out/'sim')],capture_output=True,text=True,check=True)
    result=dict(passed=True,scope='reset_toggle_primitives',physical_signoff=False,structures=structures,
        simulation=json.loads(dynamic.stdout),rtl_sha256=hashlib.sha256(rtl.read_bytes()).hexdigest(),
        simulator_sha256=hashlib.sha256((out/'sim').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':check()
