#!/usr/bin/env python3
"""Derive real M4 supply straps from a research macro's actual M3 pins.

Uses the pinned VIA34 geometry, checks the pre-existing layout for clearance,
and adds matching GDS metal/cuts/labels and LEF ports. Local geometric checks
are deliberately separate from full macro DRC/LVS/PEX and characterization.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import klayout.db as kdb

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'physical'),str(ROOT/'python')]
from inspect_sram import inspect
from sram_package import load_package
from simd_eval.architecture import ARCH


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_region(top,index):
    # A Region constructed from a recursive iterator can retain a live view
    # of the layout. Materialize it before inserting the proposed geometry.
    return kdb.Region(top.begin_shapes_rec(index)).merged()


def extend(source,out):
    package=load_package(source,ARCH)
    if out.exists():raise ValueError('preserve existing package; choose a new --out')
    shutil.copytree(package['root'],out)
    native=package['native'];gds=out/(native+'.gds');lef=out/(native+'.lef')
    before_gds=sha(gds);before_lef=sha(lef)
    layout=kdb.Layout();layout.read(str(gds));top=layout.cell(native);dbu=layout.dbu
    old_box=top.bbox();text=lef.read_text()
    tech=ROOT/'.deps/asap7/lef/asap7_tech_1x_201209.lef'
    via=re.search(r'\bVIA VIA34 Default\s+(.*?)\bEND VIA34',tech.read_text(),re.S)
    if not via:raise ValueError('locked VIA34 geometry missing')
    pieces={name:tuple(map(float,values.split())) for name,values in re.findall(
        r'LAYER\s+(\w+)\s*;\s*RECT\s+([^;]+);',via[1])}
    if set(pieces)!={'M3','M4','V3'}:raise ValueError('unexpected VIA34 layers')
    def box(coords):return kdb.Box(*[round(v/dbu) for v in coords])
    def region(layer):
        idx=layout.find_layer(layer,0)
        return snapshot_region(top,idx) if idx is not None else kdb.Region()
    original_m3=region(30);original_m4=region(40);original_v3=region(35)
    added_m4=kdb.Region();added_v3=kdb.Region();ports=[]
    width=old_box.right*dbu
    for name in ('vdd','vss'):
        pin=re.search(r'\bPIN\s+'+name+r'\s+(.*?)\bEND\s+'+name+r'\b',text,re.S)
        if not pin:raise ValueError('supply pin absent: '+name)
        shape=re.search(r'\bLAYER M3\s*;\s*RECT\s+([^;]+);',pin[1])
        if not shape:raise ValueError('expected M3 supply pin: '+name)
        x0,y0,x1,y1=map(float,shape[1].split());x=(x0+x1)/2
        chosen=None
        for track in range(math.ceil((y0+0.020-0.003)/0.048),math.floor((y1-0.020-0.003)/0.048)+1):
            y=0.003+track*0.048
            strap=box((0,y-0.012,width,y+0.012))
            m3=box(tuple(v+(x if i%2==0 else y) for i,v in enumerate(pieces['M3'])))
            cut=box(tuple(v+(x if i%2==0 else y) for i,v in enumerate(pieces['V3'])))
            # The full lower enclosure must already belong to the supply pin;
            # no new M3 shape is drawn into an unknown internal net.
            if not(kdb.Region(m3)-kdb.Region(box((x0,y0,x1,y1)))).is_empty():continue
            if not(kdb.Region(m3)-original_m3).is_empty():continue
            if not(kdb.Region(strap).sized(round(0.048/dbu)) & (original_m4+added_m4)).is_empty():continue
            if not(kdb.Region(cut).sized(round(0.024/dbu)) & (original_v3+added_v3)).is_empty():continue
            chosen=(x,y,strap,cut);break
        if chosen is None:raise ValueError('no clear, on-grid M4 access track for '+name)
        x,y,strap,cut=chosen
        # VIA34's M4 enclosure is entirely inside the actual full-width strap.
        upper=box(tuple(v+(x if i%2==0 else y)for i,v in enumerate(pieces['M4'])))
        if not(kdb.Region(upper)-kdb.Region(strap)).is_empty():raise ValueError('VIA34 upper enclosure is not covered')
        top.shapes(layout.layer(40,0)).insert(strap)
        top.shapes(layout.layer(35,0)).insert(cut)
        top.shapes(layout.layer(40,251)).insert(kdb.Text(name,kdb.Trans(round(width/2/dbu),round(y/dbu))))
        added_m4+=kdb.Region(strap);added_v3+=kdb.Region(cut)
        rectangle=(0.0,y-0.012,width,y+0.012)
        new_port='''    PORT
      LAYER M4 ;
        RECT '''+' '.join(f'{v:.6f}'for v in rectangle)+''' ;
    END
'''
        end=pin.end()-len('END '+name)
        text=text[:end]+new_port+text[end:]
        ports.append(dict(net=name,via_center_um=[x,y],m4_rectangle_um=list(rectangle)))
    if top.bbox()!=old_box:raise ValueError('supply extension changed the macro footprint')
    # Verify exact layer-region deltas before saving. Devices, cell hierarchy,
    # existing signal routing and the original M3 geometry remain unchanged.
    if not(region(30)^original_m3).is_empty():raise ValueError('M3 geometry changed')
    if not((region(40)-original_m4)^added_m4).is_empty():raise ValueError('unexpected M4 geometry delta')
    if not((region(35)-original_v3)^added_v3).is_empty():raise ValueError('unexpected V3 geometry delta')
    layout.write(str(gds));lef.write_text(text)
    inspection=inspect(out,native)
    report=dict(kind='actual_gds_supply_access_extension',
        parent_package_manifest_sha256=package['manifest_sha256'],
        parent_gds_sha256=before_gds,parent_lef_sha256=before_lef,
        derived_gds_sha256=sha(gds),derived_lef_sha256=sha(lef),
        tech_lef_sha256=sha(tech),via='VIA34',via_rectangles_um=pieces,ports=ports,
        prior_m3_geometry_preserved=True,added_only_m4_straps_and_v3_cuts=True,
        existing_m4_clearance_um=0.048,existing_v3_clearance_um=0.024,
        gds_bitcells=inspection['gds_bitcells'],physical_certified=False,
        full_drc_lvs_pex_verified=False,frequency_certified=False,
        driver_sha256=sha(__file__),
        note='Local geometry and view checks only. Parent controller timing failure and estimated Liberty remain; transistor behavior and full rule/connectivity checks are outstanding.')
    (out/'power_access.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,out/'extend_sram_power.py')
    manifest=json.loads((out/'package.json').read_text())
    names=set(manifest['views_sha256'])|{'power_access.json','extend_sram_power.py'}
    manifest.update(parent_package_manifest_sha256=package['manifest_sha256'],
        power_access_extension='Actual M4 supply straps and VIA34 cuts, matching LEF M4 ports',
        views_sha256={name:sha(out/name)for name in sorted(names)},
        physical_certified=False,frequency_certified=False,
        note=manifest['note']+' Derived package adds physical supply access; retains all unresolved qualification items.')
    (out/'package.json').write_text(json.dumps(manifest,indent=2)+'\n')
    load_package(out,ARCH)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();extend(args.source.resolve(),args.out.resolve())
