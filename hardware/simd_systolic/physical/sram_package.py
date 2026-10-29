"""Authenticated research-package views and actual generated-macro inventory."""
import hashlib
import json
import re
from pathlib import Path


def load_package(root,architecture):
    root=Path(root).resolve();record=root/'package.json'
    data=json.loads(record.read_text());native=data['native_macro']
    if data.get('kind')!='generated_asap7_6t_research_macro' or not re.fullmatch(r'[A-Za-z_]\w*',native):
        raise ValueError('unsupported SRAM package type')
    required={native+'.'+suffix for suffix in ('gds','lef','lib','sp')}|{
        data['rtl_wrapper'],'inspection.json','model_verification.json','normalization.json'}
    if not required<=data['views_sha256'].keys():raise ValueError('SRAM package is missing required authenticated views')
    if data.get('architecture_sha256')!=architecture['architecture_sha256']:raise ValueError('SRAM architecture identity changed')
    cfg=data['configuration']
    if (cfg['logical_depth'],cfg['logical_width'],cfg['memory_period_ps'])!=(
        architecture['sram_macro_depth'],architecture['sram_macro_width'],architecture['memory_period_ps']):
        raise ValueError('SRAM package differs from the architecture')
    for name,digest in data['views_sha256'].items():
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('SRAM package view changed: '+name)
    if not data.get('functional_model_checked'):raise ValueError('SRAM interface-model test missing')
    inspect=json.loads((root/'inspection.json').read_text())
    if inspect['gds_bitcells']!=65536 or inspect['schematic_bitcells']!=65536 or not inspect['pin_sets_match']:
        raise ValueError('SRAM view geometry/schematic inspection incomplete')
    model=json.loads((root/'model_verification.json').read_text())
    if not model['passed'] or model['wrapper_sha256']!=hashlib.sha256((root/data['rtl_wrapper']).read_bytes()).hexdigest():
        raise ValueError('SRAM model proof differs from its wrapper')
    bank=Path(__file__).resolve().parents[1]/'rtl/SramBank.sv'
    if model['bank_adapter_sha256']!=hashlib.sha256(bank.read_bytes()).hexdigest():raise ValueError('SRAM bank adapter changed after model verification')
    # This package has estimated timing and incomplete internal verification.
    # A user-supplied pass flag cannot turn this import path into signoff.
    return dict(root=root,native=native,liberty=root/(native+'.lib'),lef=root/(native+'.lef'),
        wrapper=root/data['rtl_wrapper'],manifest_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
        physical_certified=False,corner_matched=False,kind=data['kind'])


def inventory(netlist,native):
    netlist=Path(netlist);raw=netlist.read_text()
    instances=re.findall(r'\b'+re.escape(native)+r'\s+(\\\S+|[\w.$]+)\s*\(',raw)
    counts={name:0 for name in ('a','b','c','aggregates')};unknown=[]
    for instance in instances:
        match=re.search(r'(?:^|\.)(a|b|c|aggregates)\.depthBanks\[',instance.lstrip('\\'))
        if match:counts[match[1]]+=1
        else:unknown.append(instance)
    expected=dict(a=2,b=4,c=2,aggregates=8)
    retained={name+'_ext':count*8192 for name,count in counts.items()}
    return dict(kind='hard_sram_macro_inventory',macro=native,instances=instances,macro_counts=counts,
        unknown_instances=unknown,retained_bytes=retained,retained_bits={k:v*8 for k,v in retained.items()},
        nominal_bytes={k+'_ext':v*8192 for k,v in expected.items()},
        all_nominal_bits_retained=counts==expected and not unknown,physical_macro_certified=False,
        netlist=str(netlist),netlist_sha256=hashlib.sha256(netlist.read_bytes()).hexdigest(),
        method='actual generated 1024x64 macro instances; package separately checks GDS/schematic bitcell counts')
