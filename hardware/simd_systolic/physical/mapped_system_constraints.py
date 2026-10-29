"""Production Gray-bus constraints, including mapped flop-output inversion.

The direct-flop parser in system_constraints.py remains unchanged for the
frozen verification fixtures. Physical builds use this mapped-netlist resolver.
"""
import hashlib
import json
import re
from pathlib import Path


def canonical(value):
    return re.sub(r'[\\\s]','',value)


def crossings(netlist):
    text=Path(netlist).read_text();parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for left,right in re.findall(r'\bassign\s+([^;=]+)=([^;]+);',text):
        a,b=canonical(left),canonical(right)
        if re.fullmatch(r'[\w.$]+(?:\[\d+\])?',a) and re.fullmatch(r'[\w.$]+(?:\[\d+\])?',b):parent[find(a)]=find(b)
    cells=[]
    for match in re.finditer(r'\b(\w+)\s+(\\\S+|[\w$]+)\s*\((.*?)\);',text,re.S):
        ports={p:canonical(n)for p,n in re.findall(r'\.(\w+)\s*\(([^()]*)\)',match[3])}
        cells.append((match[1],match[2].lstrip('\\'),ports))
    qdrivers={};drivers={}
    for typ,inst,ports in cells:
        for port in ('Q','QN'):
            if port in ports and ('DFF' in typ or 'SDFF' in typ):qdrivers[find(ports[port])]=(inst+'/'+port,ports)
        for port in ('Y','Z','ZN'):
            if port in ports:drivers[find(ports[port])]=(typ,inst,ports)
    def flop(net,depth=0):
        key=find(net)
        if key in qdrivers:return qdrivers[key]
        if depth>3 or key not in drivers:raise ValueError('Gray source does not resolve to a launch flop: '+net)
        typ,inst,ports=drivers[key]
        inputs=[n for p,n in ports.items()if p not in ('Y','Z','ZN')]
        if len(inputs)!=1:raise ValueError('Gray source must be a registered bit: '+net)
        return flop(inputs[0],depth+1)
    records=[]
    # Kept stage-1/stage-2 output nets survive mapping, unlike source aliases.
    for net in list(parent)+[n for _,_,ports in cells for n in ports.values()]:
        match=re.fullmatch(r'(.*)\.(wg1|rg1)\[(\d+)\]',net)
        if not match:continue
        prefix,name,index=match.groups();first=flop(net)
        second=flop(prefix+'.'+name[:2]+'2['+index+']')
        if not first or not second:raise ValueError('missing kept synchronizer stage: '+net)
        record=dict(fifo=prefix,direction=name[:2],bit=int(index),launch=flop(first[1]['D'])[0],
                    first=first[0].rsplit('/',1)[0]+'/D',second=second[0].rsplit('/',1)[0]+'/D')
        if record not in records:records.append(record)
    if not records:raise ValueError('no mapped Gray synchronizers; refusing an unconstrained system')
    fifos={r['fifo']for r in records}
    if len(fifos)!=8:raise ValueError(f'expected 8 CDC FIFOs, found {len(fifos)}')
    for fifo in fifos:
        for direction in ('wg','rg'):
            indices=sorted(r['bit']for r in records if r['fifo']==fifo and r['direction']==direction)
            if not indices or indices!=list(range(len(indices))):raise ValueError('incomplete Gray bus: '+fifo)
    return sorted(records,key=lambda r:(r['fifo'],r['direction'],r['bit']))


def system_sdc(netlist,helper):
    records=crossings(netlist)
    lines=['set_cmd_units -time ns -capacitance fF',
        'create_clock -name core -period 1.000000 [get_ports io_coreClock]',
        'create_clock -name memory -period 2.000000 [get_ports io_memoryClock]',
        'set_clock_uncertainty 0.050 [all_clocks]',
        'set data_inputs [all_inputs]',
        'foreach excluded [get_ports {io_coreClock io_memoryClock io_coreReset}] {',
        '  set data_inputs [lsearch -all -inline -not -exact $data_inputs $excluded]',
        '}',
        'set_input_delay 0.100 -clock memory $data_inputs',
        'set_input_delay 0.100 -clock core [get_ports io_coreReset]',
        'set_input_transition 0.050 $data_inputs',
        'set_input_transition 0.050 [get_ports io_coreReset]',
        'set memory_outputs [lsearch -all -inline -not -exact [all_outputs] [get_ports io_coreActive]]',
        'set_output_delay 0.100 -clock memory $memory_outputs',
        'set_output_delay 0.100 -clock core [get_ports io_coreActive]',
        'set_load 3.898 [all_outputs]',f'source {{{helper}}}']
    # Retain FIFO implementation boundaries and constrain each Gray launch to
    # the first stage. No clock-group exception masks the max-delay check.
    reverse=('results','dones','auxResults','seedAcks')
    for fifo in sorted({r['fifo']for r in records}):
        write_core=fifo.split('.')[-1] in reverse
        for direction in ('wg','rg'):
            bits=[r for r in records if r['fifo']==fifo and r['direction']==direction]
            source_period=1 if (write_core==(direction=='wg')) else 2
            def pins(key):return '[get_pins {'+' '.join(r[key]for r in bits)+'}]'
            lines.append('simd_constrain_gray '+pins('launch')+' '+pins('first')+' '+pins('second')+f' {source_period}.000000')
    return '\n'.join(lines)+'\n'


def write_system_sdc(netlist,path,helper):
    Path(path).write_text(system_sdc(netlist,helper))
    records=crossings(netlist)
    Path(path).with_suffix('.cdc.json').write_text(json.dumps(dict(
        netlist_sha256=hashlib.sha256(Path(netlist).read_bytes()).hexdigest(),
        crossings=records,physical_signoff=False),indent=2)+'\n')
