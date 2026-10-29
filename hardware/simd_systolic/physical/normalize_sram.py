#!/usr/bin/env python3
"""Put generated macro geometry and abstract in the same full-geometry box."""
import hashlib
import json
import re
import sys
from pathlib import Path
import klayout.db as kdb


def normalize(root,native):
    root=Path(root);gds=root/(native+'.gds');lef=root/(native+'.lef');lib=root/(native+'.lib')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={p.name:digest(p)for p in (gds,lef,lib)}
    layout=kdb.Layout();layout.read(str(gds));top=layout.cell(native);box=top.bbox()
    dx,dy=-box.left*layout.dbu,-box.bottom*layout.dbu
    width,height=box.width()*layout.dbu,box.height()*layout.dbu
    text=lef.read_text();foreign=re.search(r'\bFOREIGN\s+'+re.escape(native)+r'\s+([-\d.]+)\s+([-\d.]+)\s*;',text)
    if not foreign:raise ValueError('LEF does not declare the GDS-to-abstract origin')
    ax,ay=dx-float(foreign[1]),dy-float(foreign[2])
    top.transform(kdb.Trans(-box.left,-box.bottom))
    # BOUNDARY is abstract geometry. Expand it to contain the fin/gate
    # overhang too; the device and routing geometry changes only by translation.
    boundary=layout.find_layer(100,0)
    if boundary is not None:
        top.shapes(boundary).clear();top.shapes(boundary).insert(kdb.Box(0,0,box.width(),box.height()))
    layout.write(str(gds))
    text=re.sub(r'\bFOREIGN\s+'+re.escape(native)+r'\s+[-\d.]+\s+[-\d.]+\s*;',f'FOREIGN {native} 0 0 ;',text)
    text=re.sub(r'\bSIZE\s+[-\d.]+\s+BY\s+[-\d.]+\s*;',f'SIZE {width:.6f} BY {height:.6f} ;',text)
    def rect(match):
        v=list(map(float,match.groups()));v=[v[0]+ax,v[1]+ay,v[2]+ax,v[3]+ay]
        return 'RECT '+' '.join(f'{x:.6f}'for x in v)+' ;'
    text=re.sub(r'\bRECT\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*;',rect,text)
    lef.write_text(text)
    raw=lib.read_text();raw,count=re.subn(r'\barea\s*:\s*[\d.eE+-]+\s*;',f'area : {width*height:.9f};',raw,count=1)
    if count!=1:raise ValueError('estimated Liberty has no macro area')
    lib.write_text(raw)
    report=dict(original_sha256=before,translated_gds_um=[dx,dy],translated_abstract_um=[ax,ay],
        width_um=width,height_um=height,boundary_encloses_fin_gate_overhang=True,
        topology_changed=False,timing_characterized=False)
    (root/'normalization.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':normalize(sys.argv[1],sys.argv[2])
