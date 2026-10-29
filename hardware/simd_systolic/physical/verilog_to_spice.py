#!/usr/bin/env python3
"""Convert mapped scalar cells to CDL using their actual transistor-library pin order.

Yosys resolves Verilog buses and aliases. No transistor or missing pin is invented.
This produces a schematic reference, not an LVS or extracted-parasitic result.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PHYSICAL_ONLY={'FILLER_ASAP7_75t_R','FILLERxp5_ASAP7_75t_R','TAPCELL_ASAP7_75t_R'}


def cdl_ports(text):
    logical=[]
    for line in text.splitlines():
        if line.lstrip().startswith('+'):
            if not logical:raise ValueError('orphan CDL continuation')
            logical[-1]+=' '+line.lstrip()[1:]
        else:logical.append(line)
    result={}
    for line in logical:
        fields=line.split()
        if fields and fields[0].lower()=='.subckt':
            if fields[1] in result:raise ValueError('duplicate CDL cell '+fields[1])
            result[fields[1]]=fields[2:]
    if not result:raise ValueError('empty transistor library')
    return result


def convert(verilog,cdl,out,yosys='yosys',top='ctrl_decode'):
    verilog,cdl,out=map(lambda p:Path(p).resolve(),(verilog,cdl,out))
    raw=verilog.read_text();definitions=cdl_ports(cdl.read_text())
    # Physical clock-load buffers deliberately leave their output floating.
    # Confirm the direction in the matching functional library before emitting
    # a private open node; a missing input remains an error.
    libraries='\n'.join(p.read_text()for p in sorted((cdl.parent.parent/'lib').glob('*.lib')))
    def output_pin(typ,pin):
        start=re.search(r'\bcell\s*\(\s*"?'+re.escape(typ)+r'"?\s*\)\s*\{',libraries)
        if not start:return False
        from asap7_platform import block_end
        cell=libraries[start.start():block_end(libraries,start.end()-1)]
        return re.search(r'\bpin\s*\(\s*"?'+re.escape(pin)+r'"?\s*\)\s*\{[^{}]*\bdirection\s*:\s*output\s*;',cell) is not None
    out.parent.mkdir(parents=True,exist_ok=True)
    def quoted(path):return '"'+str(path).replace('\\','\\\\').replace('"','\\"')+'"'
    with tempfile.TemporaryDirectory(prefix='cdl_convert_',dir=out.parent)as directory:
        work=Path(directory);models=work/'cells.v'
        models.write_text('\n'.join('(* blackbox *) module '+name+'('+','.join(pins)+');\ninout '+','.join(pins)+';\nendmodule'
            for name,pins in definitions.items())+'\n'+'\n'.join(
                '(* blackbox *) module '+name+'(); endmodule'for name in PHYSICAL_ONLY-definitions.keys()))
        script=work/'read.ys';net=work/'netlist.json'
        script.write_text('read_verilog -lib '+quoted(models)+'\nread_verilog '+quoted(verilog)+
            '\nhierarchy -check -top '+top+'\nwrite_json '+quoted(net)+'\n')
        with out.with_suffix('.yosys.log').open('w')as log:
            subprocess.run([yosys,'-Q','-T','-s',str(script)],stdout=log,stderr=subprocess.STDOUT,check=True)
        module=json.loads(net.read_text())['modules'][top]
    vector_names=set(re.findall(r'\b(?:input|output|inout)\s+(?:wire\s+)?\[[^]]+\]\s+(\w+)\s*(?=[;,)])',raw))
    names={};ports=[];aliases=[]
    for name,port in module['ports'].items():
        bits=port['bits'];offset=port.get('offset',0)
        for i,bit in enumerate(bits):
            index=offset+(len(bits)-1-i if port.get('upto',0)else i)
            label=f'{name}[{index}]'if len(bits)>1 or name in vector_names else name
            ports.append(label)
            if bit in names:aliases.append((label,names[bit]))
            elif isinstance(bit,int):names[bit]=label
            elif bit in ('0','1'):aliases.append((label,'VSS'if bit=='0'else'VDD'))
            else:raise ValueError('undriven top port '+label)
    def node(bit):
        if bit=='0':return 'VSS'
        if bit=='1':return 'VDD'
        if not isinstance(bit,int):raise ValueError('unknown/high-impedance net in mapped schematic')
        return names.get(bit,'n'+str(bit))
    lines=['* Schematic CDL from actual routed gates; not extracted LVS evidence.',
        '.INCLUDE "'+str(cdl)+'"','.SUBCKT '+top+' '+' '.join(ports+['VDD','VSS'])]
    lines+=['.CONNECT '+left+' '+right for left,right in aliases]
    seen_types={};physical_only={};open_outputs=[]
    for index,(instance,cell)in enumerate(module['cells'].items()):
        typ=cell['type']
        if typ in PHYSICAL_ONLY and typ not in definitions:
            if cell['connections']:raise ValueError('physical-only cell has signal pins '+instance)
            physical_only[typ]=physical_only.get(typ,0)+1
            continue
        if typ not in definitions:raise ValueError('missing transistor definition '+typ)
        connections=cell['connections'];nodes=[]
        if set(connections)-set(definitions[typ]):raise ValueError('unknown cell pin on '+instance)
        for pin in definitions[typ]:
            if pin in ('VDD','VSS'):nodes.append(pin);continue
            bits=connections.get(pin,[])
            if not bits and output_pin(typ,pin):
                nodes.append('open_'+str(index)+'_'+pin);open_outputs.append(instance+'/'+pin);continue
            if len(bits)!=1:raise ValueError('missing/non-scalar pin '+instance+'/'+pin)
            nodes.append(node(bits[0]))
        lines.append('X'+str(index)+' '+' '.join(nodes)+' '+typ)
        seen_types[typ]=seen_types.get(typ,0)+1
    lines.append('.ENDS '+top);out.write_text('\n'.join(lines)+'\n')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(top=top,cells=len(module['cells']),electrical_cells=sum(seen_types.values()),
        omitted_empty_physical_cells=physical_only,cell_types=seen_types,port_count=len(ports),
        intentionally_open_outputs=open_outputs,
        pin_order='actual CDL .SUBCKT declarations',all_signal_pins_resolved=True,
        verilog_sha256=sha(verilog),cdl_sha256=sha(cdl),output_sha256=sha(out),
        generator_sha256=sha(__file__),lvs_passed=False,parasitics_extracted=False)
    out.with_suffix('.conversion.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verilog',type=Path,required=True)
    p.add_argument('--cdl',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--yosys',default=shutil.which('yosys'));p.add_argument('--top',default='ctrl_decode')
    args=p.parse_args();print(json.dumps(convert(args.verilog,args.cdl,args.out,args.yosys,args.top)))
