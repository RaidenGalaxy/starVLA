"""Prove constant endpoints from mapped tie-cell connectivity.

STA has no timed launch path to a D pin driven by a tie cell. Accept only
explicit LOGIC0/LOGIC1 -> optional BUF/INV/alias chains and MUX2 with a proven
constant selector and selected data input, including constant
output ports; every other coverage warning
remains a signoff failure. The original check_setup report is retained.
"""
import re
import ast
from collections import defaultdict,deque


def _blocks(text,kind):
    """Read named Liberty blocks without mistaking nested timing tables."""
    for match in re.finditer(r'\b'+kind+r'\s*\(\s*"?([\w]+)"?\s*\)\s*\{',text):
        start=pos=match.end();depth=1;quoted=False;escaped=False
        while depth and pos<len(text):
            char=text[pos];pos+=1
            if escaped:escaped=False;continue
            if quoted and char=='\\':escaped=True;continue
            if char=='"':quoted=not quoted
            elif not quoted:
                if char=='{':depth+=1
                elif char=='}':depth-=1
        if depth:raise ValueError('unterminated Liberty '+kind+' block')
        yield match[1],text[start:pos-1]


def _logic_functions(library):
    functions={}
    for kind,body in _blocks(library,'cell'):
        if re.search(r'\b(?:ff|ff_bank|latch|latch_bank)\s*\(',body):continue
        pins=dict(_blocks(body,'pin'))
        inputs={name for name,pin in pins.items()if re.search(r'\bdirection\s*:\s*input\s*;',pin)}
        outputs={}
        for name,pin in pins.items():
            expression=re.search(r'\bfunction\s*:\s*"([^"]+)"\s*;',pin)
            if not expression or not re.search(r'\bdirection\s*:\s*output\s*;',pin):continue
            try:tree=ast.parse(expression[1].replace('!','~').replace('*','&').replace('+','|'),mode='eval').body
            except SyntaxError:continue
            allowed=(ast.Expression,ast.Name,ast.Load,ast.Constant,ast.UnaryOp,ast.Invert,ast.BinOp,ast.BitAnd,ast.BitOr,ast.BitXor)
            if any(not isinstance(n,allowed)or isinstance(n,ast.Constant)and n.value not in(0,1)for n in ast.walk(tree)):continue
            variables={n.id for n in ast.walk(tree)if isinstance(n,ast.Name)}
            if variables<=inputs:outputs[name]=(expression[1],tree,variables)
        if outputs:functions[kind]=outputs
    return functions


def _evaluate(tree,values):
    if isinstance(tree,ast.Name):return values[tree.id]
    if isinstance(tree,ast.Constant):return tree.value
    if isinstance(tree,ast.UnaryOp):return _evaluate(tree.operand,values)^1
    left,right=_evaluate(tree.left,values),_evaluate(tree.right,values)
    if isinstance(tree.op,ast.BitAnd):return left&right
    if isinstance(tree.op,ast.BitOr):return left|right
    return left^right


def analyze(netlist, report,library=None):
    canonical=lambda s:re.sub(r'\s+','',s)
    cells={};constants={};outputs=set();mux_proofs={}
    for match in re.finditer(r'\boutput\s+(?:(?:wire|logic|reg)\s+)?(?:\[(\d+)\s*:\s*(\d+)\]\s*)?([^;]+);',netlist):
        hi,lo,names=match.groups()
        for name in names.split(','):
            name=canonical(name).removeprefix('\\')
            if not re.fullmatch(r'[\w.$]+',name):continue
            outputs.update(f'{name}[{i}]'for i in range(min(int(hi),int(lo)),max(int(hi),int(lo))+1))if hi else outputs.add(name)
    aliases=[]
    for left,right in re.findall(r'\bassign\s+([^;=]+)=([^;=]+);',netlist):
        left,right=canonical(left),canonical(right)
        if all(re.fullmatch(r'(?:\\[^\s{},]+|[\w.$]+(?:\[\d+\])?)',name)for name in(left,right)):
            aliases.append((left,right))
    for match in re.finditer(r'^\s*(\w+)\s+(\S+)\s*\((.*?)\);',netlist,re.M|re.S):
        kind,name,body=match.groups();name=name.removeprefix('\\')
        ports={p:canonical(n)for p,n in re.findall(r'\.(\w+)\s*\(([^()]*)\)',body)}
        cells[name]=(kind,ports)
        if kind in ('LOGIC0_X1','LOGIC1_X1')and 'Z'in ports:
            constants[ports['Z']]=(int(kind=='LOGIC1_X1'),name)
        if kind=='TIEHIx1_ASAP7_75t_R' and 'H'in ports:constants[ports['H']]=(1,name)
        if kind=='TIELOx1_ASAP7_75t_R' and 'L'in ports:constants[ports['L']]=(0,name)
    # Routed hold repair can introduce long buffer chains in reverse netlist
    # order. Repeated whole-design scans then cost O(cells * chain depth).
    # Visit only consumers of newly proven constants. Keep exactly the same
    # accepted cells and MUX proof rule; sequential cells never propagate.
    consumers=defaultdict(list);alias_consumers=defaultdict(list)
    logic_cells={};logic_proofs={};functions=_logic_functions(library)if library is not None else{}
    for name,(kind,ports)in cells.items():
        inputs=('S','A','B')if kind=='MUX2_X1'else('A',)if re.fullmatch(r'(BUF|CLKBUF|INV)_X\d+',kind)else()
        for pin in inputs:
            if pin in ports:consumers[ports[pin]].append(name)
        if inputs:continue
        for output,(expression,tree,variables)in functions.get(kind,{}).items():
            if output not in ports or not variables or not variables<=ports.keys():continue
            key=(name,output);logic_cells[key]=(expression,tree,variables)
            for pin in variables:consumers[ports[pin]].append(key)
    for left,right in aliases:alias_consumers[right].append(left)
    todo=deque(constants)
    while todo:
        net=todo.popleft()
        for name in consumers.get(net,()):
            if isinstance(name,tuple):
                cell,pin=name;kind,ports=cells[cell];output=ports[pin]
                expression,tree,variables=logic_cells[name]
                if output in constants or any(ports[p]not in constants for p in variables):continue
                values={p:constants[ports[p]][0]for p in variables}
                value=_evaluate(tree,values)
                first=sorted(variables)[0];constants[output]=(value,constants[ports[first]][1])
                logic_proofs[output]=dict(cell=cell,cell_type=kind,output_pin=pin,function=expression,value=value,
                    inputs={p:dict(net=ports[p],value=values[p],tie_cell=constants[ports[p]][1])for p in sorted(variables)})
                todo.append(output);continue
            kind,ports=cells[name]
            if kind=='MUX2_X1':
                if ports.get('S')not in constants:continue
                select,selector_tie=constants[ports['S']]
                chosen=ports.get('B'if select else'A');output=ports.get('Z')
                if chosen not in constants or not output or output in constants:continue
                constants[output]=constants[chosen]
                mux_proofs[output]=dict(cell=name,selector_net=ports['S'],selector_value=select,
                    selector_tie_cell=selector_tie,selected_data_net=chosen)
            else:
                if ports.get('A')not in constants:continue
                output=ports.get('ZN'if kind.startswith('INV_')else'Z')
                if not output or output in constants:continue
                value,source=constants[ports['A']]
                constants[output]=(value^int(kind.startswith('INV_')),source)
            todo.append(output)
        for left in alias_consumers.get(net,()):
            if left not in constants:
                constants[left]=constants[net];todo.append(left)
    lines=[line.strip()for line in report.splitlines()if line.strip()]
    if not lines:return {'clean':True,'constant_endpoints':[],'unproven':[]}
    header=re.fullmatch(r'Warning: There (?:are|is) (\d+) unconstrained endpoints?\.',lines[0])
    if not header or int(header[1])!=len(lines)-1:
        return {'clean':False,'constant_endpoints':[],'unproven':lines}
    proved=[];unknown=[]
    for pin in lines[1:]:
        if '/'not in pin:
            net=canonical(pin)
            if net.removeprefix('\\')not in outputs or net not in constants:unknown.append(pin);continue
            value,source=constants[net]
            proved.append({'pin':pin,'net':net,'value':value,'tie_cell':source});continue
        name,port=pin.rsplit('/',1);kind,ports=cells.get(name.removeprefix('\\'),('',{}));net=ports.get(port)
        if net not in constants:unknown.append(pin);continue
        value,source=constants[net]
        proved.append({'pin':pin,'net':net,'value':value,'tie_cell':source})
    result={'clean':not unknown,'constant_endpoints':proved,'unproven':unknown,
            'constant_mux_proofs':mux_proofs}
    if library is not None:result['constant_logic_proofs']=logic_proofs
    return result
