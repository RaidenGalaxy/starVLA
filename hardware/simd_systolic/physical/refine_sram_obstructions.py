#!/usr/bin/env python3
"""Conservative upper-metal LEF obstructions derived from actual GDS.

Blanket M4/M5 obstructions hide all power access even when the array has no
metal on those layers. Every existing polygon remains covered by a real pin
or an obstruction; lower device/routing layers keep their blanket protection.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import klayout.db as kdb

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'physical'),str(ROOT/'python')]
from sram_package import load_package
from simd_eval.architecture import ARCH
from inspect_sram import inspect


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refine(source,out):
    package=load_package(source,ARCH)
    if out.exists():raise ValueError('preserve existing package; choose a new --out')
    shutil.copytree(package['root'],out)
    native=package['native'];lef=out/(native+'.lef');text=lef.read_text();old_sha=sha(lef)
    layout=kdb.Layout();layout.read(str(out/(native+'.gds')));top=layout.cell(native);dbu=layout.dbu
    obs=re.search(r'\bOBS\s+(.*?)\n\s*END(?=\s*\n\s*END '+re.escape(native)+r')',text,re.S)
    if not obs:raise ValueError('macro obstruction section missing')
    lower=[];evidence=[];body=obs[1]
    for layer,content in re.findall(r'\bLAYER\s+(\w+)\s*;(.*?)(?=\bLAYER\s+\w+\s*;|$)',body,re.S):
        if layer not in ('M4','M5'):lower.append('    LAYER '+layer+' ;'+content.rstrip()+'\n')
    upper=[]
    for name,number in [('M4',40),('M5',50)]:
        index=layout.find_layer(number,0)
        geometry=kdb.Region(top.begin_shapes_rec(index)).merged() if index is not None else kdb.Region()
        pins=kdb.Region()
        for pin in re.finditer(r'\bPIN\s+(\S+)\s+(.*?)\bEND\s+\1(?=\s|$)',text,re.S):
            if pin[1] not in ('vdd','vss'):continue
            for layer,content in re.findall(r'\bLAYER\s+(\w+)\s*;(.*?)(?=\bLAYER\s+\w+\s*;|\bEND|$)',pin[2],re.S):
                if layer!=name:continue
                for row in re.findall(r'\bRECT\s+([^;]+);',content):
                    pins.insert(kdb.Box(*[round(float(v)/dbu)for v in row.split()]))
        remainder=(geometry-pins).merged()
        conservative=kdb.Region()
        for polygon in remainder.each():conservative.insert(polygon.bbox())
        conservative=conservative.merged()
        # Polygon bounding boxes may cover pin ports; cut those ports out
        # again. Their own LEF PIN shapes continue to reserve that metal.
        conservative=(conservative-pins).merged()
        uncovered=geometry-(conservative+pins)
        if not uncovered.is_empty():raise ValueError('real routing left unprotected: '+name)
        upper.append('    LAYER '+name+' ;\n')
        # LEF supports rectangular OBS. Decompose without losing coverage.
        rectangles=[]
        for polygon in conservative.each():
            rectangles.extend(kdb.Region(polygon).decompose_trapezoids_to_region().each())
        # The rectilinear GDS input produces rectangles after trapezoid
        # decomposition; refuse a slanted shape instead of widening a hole.
        final=kdb.Region()
        for polygon in rectangles:
            rectangle=polygon.bbox()
            if not(kdb.Region(polygon)^kdb.Region(rectangle)).is_empty():raise ValueError('nonrectangular routing obstruction')
            final.insert(rectangle)
            upper.append('      RECT '+' '.join(f'{v*dbu:.6f}'for v in
                (rectangle.left,rectangle.bottom,rectangle.right,rectangle.top))+' ;\n')
        if not(geometry-(final+pins)).is_empty():raise ValueError('rectangular abstract lost routing coverage')
        evidence.append(dict(layer=name,rectangles=len(rectangles),
            actual_metal_area_um2=geometry.area()*dbu*dbu,
            obstruction_area_um2=final.area()*dbu*dbu,all_actual_metal_protected=True))
    replacement='OBS\n'+''.join(lower+upper)+'  END'
    text=text[:obs.start()]+replacement+text[obs.end():];lef.write_text(text)
    inspect(out,native)
    report=dict(kind='gds_derived_conservative_upper_metal_abstract',
        parent_package_manifest_sha256=package['manifest_sha256'],parent_lef_sha256=old_sha,
        derived_lef_sha256=sha(lef),gds_sha256=sha(out/(native+'.gds')),
        gds_geometry_changed=False,lower_layer_obstructions_preserved=True,layers=evidence,
        full_drc_lvs_pex_verified=False,frequency_certified=False,driver_sha256=sha(__file__))
    (out/'obstruction_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,out/'refine_sram_obstructions.py')
    manifest=json.loads((out/'package.json').read_text())
    files=set(manifest['views_sha256'])|{'obstruction_refinement.json','refine_sram_obstructions.py'}
    manifest.update(obstruction_parent_package_manifest_sha256=package['manifest_sha256'],
        views_sha256={n:sha(out/n)for n in sorted(files)},physical_certified=False,frequency_certified=False)
    (out/'package.json').write_text(json.dumps(manifest,indent=2)+'\n')
    load_package(out,ARCH);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();refine(args.source.resolve(),args.out.resolve())
