#!/usr/bin/env python3
"""Check elaborated synchronizer connectivity; this is not physical CDC signoff."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def check():
    out=ROOT/'.build/cdc';out.mkdir(parents=True,exist_ok=True)
    rtl=ROOT/'rtl/AsyncFifo.sv';netlist=out/'structure.json'
    command=f'read_verilog -sv {rtl}; hierarchy -top AsyncFifo -chparam WIDTH 32 -chparam ADDR 4; proc; opt; write_json {netlist}'
    proc=subprocess.run(['yosys','-Q','-T','-p',command],check=True,capture_output=True,text=True)
    (out/'structure.log').write_text(proc.stdout+proc.stderr)
    m=json.loads(netlist.read_text())['modules']['AsyncFifo'];nets=m['netnames'];cells=m['cells'];proof=[]
    def driver(name):
        bits=nets[name]['bits'];found=[(n,c)for n,c in cells.items()if c['connections'].get('Q')==bits]
        assert len(found)==1,(name,found)
        return found[0]
    for source,first,second,srcclock,dstclock,reset in [('wg','wg1','wg2','wclk','rclk','rrst'),('rg','rg1','rg2','rclk','wclk','wrst')]:
        sn,sc=driver(source);fn,fc=driver(first);tn,tc=driver(second)
        assert sc['connections']['CLK']==nets[srcclock]['bits']
        for c in (fc,tc):
            assert c['type']=='$adff' and c['connections']['CLK']==nets[dstclock]['bits']
            assert c['connections']['ARST']==nets[reset]['bits']
            assert int(c['parameters']['ARST_POLARITY'],2)==1 and int(c['parameters']['ARST_VALUE'],2)==0
        assert fc['connections']['D']==nets[source]['bits'] and tc['connections']['D']==nets[first]['bits']
        assert nets[first]['attributes'].get('ASYNC_REG')=='TRUE' and nets[second]['attributes'].get('ASYNC_REG')=='TRUE'
        for cellname,cell in cells.items():
            for port,bits in cell['connections'].items():
                if cell['port_directions'][port]!='input':continue
                if set(bits)&set(nets[first]['bits']):assert cellname==tn and port=='D','first synchronizer stage drives functional logic'
                if set(bits)&set(nets[source]['bits']):assert cellname==fn and port=='D','raw Gray pointer bypasses synchronizers'
        proof.append(dict(source=source,first=first,second=second,source_clock=srcclock,destination_clock=dstclock,
                          reset=reset,bits=len(nets[source]['bits'])))
    sim=subprocess.run([str(out/'sim')],check=True,capture_output=True,text=True)
    dynamic=json.loads(sim.stdout)
    result=dict(passed=True,scope='standalone_fifo_structure_and_simulation',physical_signoff=False,
                synchronizers=proof,simulation=dynamic,rtl_sha256=hashlib.sha256(rtl.read_bytes()).hexdigest(),
                simulator_sha256=hashlib.sha256((out/'sim').read_bytes()).hexdigest(),
                netlist_sha256=hashlib.sha256(netlist.read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':check()
