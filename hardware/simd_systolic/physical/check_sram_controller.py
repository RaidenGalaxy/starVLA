#!/usr/bin/env python3
"""Check placed controller cell counts and geometry, not routed connectivity."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
import klayout.db as kdb


def audit(gds,netlist,library,top='ctrl_decode'):
    dut=kdb.Layout();dut.read(str(gds));reference=kdb.Layout();reference.read(str(library))
    root=dut.cell(top)
    if root is None:raise ValueError('controller root is missing')
    expected=collections.Counter(typ for typ,instance in re.findall(
        r'\b(\w+)\s+(\\\S+|[\w$]+)\s*\(',Path(netlist).read_text())if typ!='module')
    actual=collections.Counter()
    for inst in root.each_inst():actual[dut.cell(inst.cell_index).name]+=inst.na*inst.nb if inst.is_regular_array()else 1
    routing_vias={name:count for name,count in actual.items()if re.fullmatch(r'VIA_VIA[1-9][1-9]',name)}
    for name in routing_vias:del actual[name]
    if not expected:raise ValueError('no controller instances found in the schematic')
    problems=[]
    if expected!=actual:problems.append('placed instance multiset differs from the routed gate netlist')
    layers={(x.layer,x.datatype)for x in dut.layer_infos()}|{(x.layer,x.datatype)for x in reference.layer_infos()}
    for name in sorted(actual):
        a,b=dut.cell(name),reference.cell(name)
        if b is None:problems.append('standard-cell reference missing: '+name);continue
        for layer,purpose in layers:
            ai,bi=dut.find_layer(layer,purpose),reference.find_layer(layer,purpose)
            ar=kdb.Region(a.begin_shapes_rec(ai))if ai is not None else kdb.Region()
            br=kdb.Region(b.begin_shapes_rec(bi))if bi is not None else kdb.Region()
            if abs(dut.dbu-reference.dbu)>1e-12:
                ratio=reference.dbu/dut.dbu
                br.transform(kdb.ICplxTrans(ratio,0,False,0,0))
            if not(ar^br).is_empty():problems.append(f'cell geometry differs: {name} layer {layer}/{purpose}');break
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return dict(passed=not problems,scope='controller instance counts and standard-cell polygon geometry',
        routed_instances=sum(expected.values()),placed_instances=sum(actual.values()),cell_types=len(actual),
        routing_via_instances=routing_vias,
        problems=problems,gds_sha256=sha(gds),netlist_sha256=sha(netlist),library_gds_sha256=sha(library),
        generator_sha256=sha(__file__),routing_connectivity_checked=False,transistor_lvs_passed=False,drc_passed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gds',type=Path,required=True)
    p.add_argument('--netlist',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    report=audit(args.gds,args.netlist,args.library);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ('passed','routed_instances','placed_instances','cell_types','problems')}))
    if not report['passed']:raise SystemExit(1)
