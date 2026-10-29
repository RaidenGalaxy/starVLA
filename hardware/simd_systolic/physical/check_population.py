#!/usr/bin/env python3
"""Audit retained recursive counter boundaries in the actual mapped mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check(directory):
    directory=directory.resolve()
    identity=json.loads((directory/'build_identity.json').read_text())
    caps=json.loads((directory/'generated/capabilities.json').read_text())
    if identity['scope']!='array'or not caps.get('balanced_event_population_count'):
        raise ValueError('an array built with balanced event counters is required')
    net=directory/'synth.v';stats=directory/'cell_statistics.json'
    if sha(net)!=identity['mapped_sha256']:raise ValueError('mapped netlist changed')
    data=json.loads(stats.read_text())
    modules={k.lstrip('\\'):v for k,v in data['modules'].items()};cache={}
    def count(module,wanted):
        key=module,wanted
        if key not in cache:
            cache[key]=sum(n if typ.lstrip('\\')==wanted else n*count(typ.lstrip('\\'),wanted)
                for typ,n in modules.get(module,{}).get('num_cells_by_type',{}).items())
        return cache[key]
    actual={}
    for name in modules:
        match=re.search(r"\\SimdPopulationCount\\N=s32'([01]{32})$",name)
        if match:actual[str(int(match[1],2))]=count(identity['top'],name)
    pes=caps['rows']*caps['cols']
    # Each PE has prefix-2/4/8/16 mask counts plus a separate 16-bit
    # overflow count. Every larger tree contains two kept child instances.
    expected={str(n):pes*per_pe for n,per_pe in((2,23),(4,11),(8,5),(16,2))}
    report=dict(passed=actual==expected,scope='mapped_population_tree_retention',
        rows=caps['rows'],cols=caps['cols'],instances_by_input_width=actual,
        expected_instances_by_input_width=expected,counter_settle_cycles=caps['counter_settle_cycles'],
        rtl_sha256=identity['rtl_sha256'],netlist_sha256=sha(net),
        source_sha256=sha(directory/'inputs/PopulationCount.sv'),
        cell_statistics_sha256=sha(stats),checker_sha256=sha(Path(__file__)),timing_validated=False,
        limitation='Mapped structure inventory only; arithmetic equivalence, gate simulation and extracted timing are separate evidence.')
    (directory/'verification_population_tree.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    if not report['passed']:raise AssertionError('recursive population-count boundaries merged or missing')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    check(parser.parse_args().directory)
