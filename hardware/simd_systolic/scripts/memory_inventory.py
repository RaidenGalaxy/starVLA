#!/usr/bin/env python3
"""Count retained standard-cell storage bits from mapped memory Q connections."""
import argparse,hashlib,json,re
from pathlib import Path

def inventory(netlist):
    macro='srambank_256x4x64_6t122'
    text=netlist.read_text()
    instances=re.findall(r'\b'+macro+r'\s+(\\?[^\s(]+)\s*\(',text)
    if instances:
        counts={name:0 for name in ('a','b','c','aggregates')}
        unknown=[]
        for instance in instances:
            name=instance.lstrip('\\')
            match=re.search(r'(?:^|\.)(a|b|c|aggregates)\.',name)
            if match:counts[match[1]]+=1
            else:unknown.append(name)
        expected=dict(a=2,b=4,c=2,aggregates=8)
        return dict(kind='hard_sram_macro_inventory',netlist=str(netlist.resolve()),
            netlist_sha256=hashlib.sha256(netlist.read_bytes()).hexdigest(),macro=macro,
            instances=instances,macro_counts=counts,unknown_instances=unknown,
            retained_bits={name+'_ext':count*65536 for name,count in counts.items()},
            retained_bytes={name+'_ext':count*8192 for name,count in counts.items()},
            nominal_bytes={name+'_ext':count*8192 for name,count in expected.items()},
            all_nominal_bits_retained=counts==expected and not unknown,
            physical_macro_certified=False,
            method='explicit 1024x64 hard-macro instances; address behavior is checked by full SRAM sweeps')
    text=netlist.read_text();banks={name:set()for name in('a_ext','b_ext','c_ext','aggregates_ext')}
    for match in re.finditer(r'\.(?:Q|QN)\(\\(?:tile\.)?(a_ext|b_ext|c_ext|aggregates_ext)\.Memory\[(\d+)\]\s*\[(\d+)\]\)',text):
        banks[match[1]].add((int(match[2]),int(match[3])))
    nominal={'a_ext':16384,'b_ext':32768,'c_ext':16384,'aggregates_ext':65536}
    return dict(kind='mapped_standard_cell_memory_inventory',netlist=str(netlist.resolve()),
        netlist_sha256=hashlib.sha256(netlist.read_bytes()).hexdigest(),
        method='distinct mapped Q/QN connections to retained named memory word bits; excludes register files and pipeline storage',
        retained_bits={name:len(bits)for name,bits in banks.items()},
        retained_bytes={name:len(bits)//8 for name,bits in banks.items()},nominal_bytes=nominal,
        all_nominal_bits_retained=all(len(banks[name])==size*8 for name,size in nominal.items()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('netlist',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();result=inventory(args.netlist)
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
