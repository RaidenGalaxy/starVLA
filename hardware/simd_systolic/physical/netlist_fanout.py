"""Census scalar port bindings in OpenROAD/Yosys gate-netlist exports."""
import re
from collections import Counter
from pathlib import Path

PORT=re.compile(r'\.(\w+)\(\s*(\\[^\s()]+|[\w$]+)(\s*\[\s*\d+\s*\])?\s*\)')


def census(path):
    references=Counter();clocks=set()
    with Path(path).open()as stream:
        for line in stream:
            if line.lstrip().startswith(('//','/*','*')):continue
            for match in PORT.finditer(line.split('//',1)[0]):
                pin,net,select=match.groups();net=net.lstrip('\\')
                if select:net+=re.sub(r'\s','',select)
                references[net]+=1
                if pin in('CLK','clk','CK','clock'):clocks.add(net)
    return references,clocks
