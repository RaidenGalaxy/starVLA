#!/usr/bin/env python3
"""Build matched GDS/schematic views with a proved, routed controller ECO.

This research package still has estimated Liberty and no full-array DRC/LVS,
transistor simulation or PVT characterization. Controller timing is a scoped
piece of evidence, never a certificate for the complete SRAM.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import klayout.db as kdb

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'physical'),str(ROOT/'python')]
from sram_package import load_package
from simd_eval.architecture import ARCH
from inspect_sram import inspect
from check_sram_controller import audit
from verilog_to_spice import convert,cdl_ports


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_evidence(data):
    for path,digest in data['evidence'].items():
        if sha(path)!=digest:raise ValueError('controller evidence changed: '+path)


def definition(raw,name):
    match=re.search(r'^\.subckt\s+'+re.escape(name)+r'\s+.*?^\.ends\s+'+re.escape(name)+r'\s*$',raw,re.M|re.I|re.S)
    if not match:raise ValueError('schematic definition missing: '+name)
    return match


def labels(layout,cell):
    result=[]
    for index in layout.layer_indexes():
        info=layout.get_info(index)
        if info.datatype!=251:continue
        for shape in cell.shapes(index).each():
            if shape.is_text():
                t=shape.text
                result.append((info.layer,t.string,round(t.trans.disp.x*layout.dbu,9),round(t.trans.disp.y*layout.dbu,9)))
    return sorted(result)


def integrate(args):
    source=load_package(args.source,ARCH);eco=args.eco.resolve();proof=args.equivalence.resolve()
    if (source['root']/'power_access.json').exists():
        raise ValueError('integrate into the raw macro, then regenerate power access against the new geometry')
    timing=json.loads((eco/'verification.json').read_text());equiv=json.loads((proof/'verification.json').read_text())
    validate_evidence(timing);validate_evidence(equiv)
    if (timing['returncode'] or timing['period_ns']!=2 or timing['mode_sdel']!=0 or
            timing['fixed_mode_setup_wns_ns']<0 or timing['fixed_mode_hold_wns_ns']<0 or
            (eco/'route.drc').read_text().strip() or 'VIOLATED'in(eco/'electrical.rpt').read_text().upper()):
        raise ValueError('controller timing/routing/electrical evidence incomplete')
    if not equiv['passed']or sha(proof/'candidate.v')!=sha(eco/'controller_eco.v'):
        raise ValueError('equivalence proof is not for this controller')
    original=json.loads((source['root']/'netlist_for_lvs.conversion.json').read_text())
    if original['verilog_sha256']!=sha(proof/'original.v'):
        raise ValueError('controller proof parent differs from the selected macro schematic')
    out=args.out.resolve()
    if out.exists():raise ValueError('preserve existing macro; choose a new --out')
    shutil.copytree(source['root'],out);work=out/'controller_eco';shutil.copytree(eco,work)
    shutil.copytree(proof,out/'controller_equivalence');shutil.copy2(__file__,out/'integration_driver.py')
    tech=ROOT/'.deps/openfinram/tech';library=tech/'gds/asap7sc7p5t_28_R_220121a.gds'
    stream=ROOT/'.deps/openfinram/scripts/def_to_gds.py';controller_gds=work/'controller_eco.gds'
    with(work/'gds_build.log').open('w')as log:
        subprocess.run([sys.executable,str(stream),'--def-file',str(work/'controller_eco.def'),
            '--tech-lef',str(tech/'lef/asap7_tech.lef'),'--cell-lef',str(tech/'lef/asap7sc7p5t_28_R.lef'),
            '--macro-gds',str(library),'--output-gds',str(controller_gds)],
            check=True,stdout=log,stderr=subprocess.STDOUT)
    geometry=audit(controller_gds,work/'controller_eco.v',library)
    if not geometry['passed']:raise ValueError('controller stream-out cell geometry mismatch')
    (work/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    gds=out/(source['native']+'.gds');layout=kdb.Layout();layout.read(str(gds))
    controller=kdb.Layout();controller.read(str(controller_gds));old=layout.cell('ctrl_decode');new=controller.cell('ctrl_decode')
    if old is None or new is None:raise ValueError('controller hierarchy missing')
    if labels(layout,old)!=labels(controller,new):raise ValueError('controller port labels moved')
    before_box=layout.cell(source['native']).dbbox()
    # Shared standard cells and routing vias must have identical geometry;
    # only the proved controller root is allowed to change on overwrite.
    for cell in controller.each_cell():
        previous=layout.cell(cell.name)
        if previous is None or cell.name=='ctrl_decode':continue
        for index in controller.layer_indexes():
            info=controller.get_info(index);other=layout.find_layer(info.layer,info.datatype)
            a=kdb.Region(cell.begin_shapes_rec(index)).merged()
            if abs(controller.dbu-layout.dbu)>1e-12:a.transform(kdb.ICplxTrans(controller.dbu/layout.dbu,0,False,0,0))
            b=kdb.Region(previous.begin_shapes_rec(other)).merged()if other is not None else kdb.Region()
            if not(a^b).is_empty():raise ValueError('shared library geometry changed: '+cell.name)
    options=kdb.LoadLayoutOptions();options.cell_conflict_resolution=kdb.LoadLayoutOptions.OverwriteCell
    layout.read(str(controller_gds),options)
    if layout.cell(source['native']).dbbox()!=before_box:raise ValueError('controller ECO changed macro footprint')
    layout.write(str(gds))
    # Reuse the original top-level controller pin order. SPICE internal nodes
    # are named, so reordering the declaration preserves all new connections.
    schematic=out/(source['native']+'.sp');raw=schematic.read_text();olddef=definition(raw,'ctrl_decode')
    cdl=tech/'cdl/asap7sc7p5t_28_R.cdl';candidate=work/'controller_eco.sp'
    convert(work/'controller_eco.v',cdl,candidate)
    newraw=candidate.read_text();newdef=definition(newraw,'ctrl_decode');ports=cdl_ports(raw)['ctrl_decode'];newports=cdl_ports(newraw)['ctrl_decode']
    if set(ports)!=set(newports):raise ValueError('controller schematic pin set changed')
    body=newdef[0].split('\n',1)[1];replacement='.SUBCKT ctrl_decode '+' '.join(ports)+'\n'+body
    raw=raw[:olddef.start()]+replacement+raw[olddef.end():]
    available=cdl_ports(raw);required=json.loads(candidate.with_suffix('.conversion.json').read_text())['cell_types']
    library_raw=cdl.read_text()
    for name in required:
        if name not in available:raw=definition(library_raw,name)[0]+'\n'+raw
    schematic.write_text(raw)
    inspection=inspect(out,source['native'])
    integrated_geometry=audit(gds,work/'controller_eco.v',library)
    if not integrated_geometry['passed']:raise ValueError('integrated controller geometry does not match its netlist')
    (work/'integrated_geometry.json').write_text(json.dumps(integrated_geometry,indent=2)+'\n')
    report=dict(kind='matched_macro_controller_eco_integration',
        parent_package_manifest_sha256=source['manifest_sha256'],controller_integrated=True,
        unchanged_controller_port_positions=True,controller_sequential_equivalence=True,
        controller_fixed_mode_setup_wns_ns=timing['fixed_mode_setup_wns_ns'],
        controller_fixed_mode_hold_wns_ns=timing['fixed_mode_hold_wns_ns'],controller_period_ns=2,
        gds_bitcells=inspection['gds_bitcells'],logical_model_unchanged=sha(out/'Asap7SramMacro.sv')==sha(source['wrapper']),
        physical_frequency_certified=False,whole_macro_drc_lvs_pex_verified=False,
        estimated_liberty_not_recharacterized=True,
        source_files_sha256={str(p):sha(p)for p in [stream,library,cdl,Path(__file__)]})
    (out/'controller_integration.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest=json.loads((out/'package.json').read_text())
    manifest.update(controller_eco_parent_package_manifest_sha256=source['manifest_sha256'],
        controller_eco_integrated=True,physical_certified=False,frequency_certified=False,
        note='Routed controller ECO integrated in GDS and schematic; fixed sdel=0 controller timing passes at TT/2ns. Parent timing files are historical. Full-array DRC/LVS/PEX, transistor behavior and PVT characterization remain incomplete; Liberty stays estimated.')
    manifest['views_sha256']={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p!=out/'package.json'}
    (out/'package.json').write_text(json.dumps(manifest,indent=2)+'\n');load_package(out,ARCH)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--eco',type=Path,required=True);p.add_argument('--equivalence',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    integrate(p.parse_args())
