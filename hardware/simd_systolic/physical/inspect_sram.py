#!/usr/bin/env python3
"""Structural checks on generated views; deliberately not a DRC/LVS claim."""
import functools
import json
import re
import sys
from pathlib import Path
import klayout.db as kdb


def inspect(root,native):
    root=Path(root);layout=kdb.Layout();layout.read(str(root/(native+'.gds')))
    top=layout.cell(native)
    if top is None:raise ValueError('native macro is missing from GDS')
    active='sram_cell_6t_122'
    @functools.lru_cache(None)
    def gds_count(index):
        cell=layout.cell(index)
        if cell.name==active:return 1
        return sum((inst.na*inst.nb if inst.is_regular_array()else 1)*gds_count(inst.cell_index)
                   for inst in cell.each_inst())
    logical=[]
    for line in (root/(native+'.sp')).read_text().splitlines():
        if line.lstrip().startswith('+'):logical[-1]+=' '+line.lstrip()[1:]
        else:logical.append(line)
    definitions={};ports={};current=None
    for line in logical:
        fields=line.split()
        if not fields or fields[0].startswith('*'):continue
        if fields[0].lower()=='.subckt':
            current=fields[1].lower();ports[current]=fields[2:]
            if current in definitions:raise ValueError('duplicate schematic subcircuit '+current)
            definitions[current]=[]
        elif fields[0].lower()=='.ends':current=None
        elif current:definitions[current].append(fields)
    if native.lower()not in definitions:raise ValueError('native macro is missing from schematic')
    @functools.lru_cache(None)
    def schematic_count(name):
        if name not in definitions:raise ValueError('unresolved schematic subcircuit '+name)
        if name==active:return 1
        return sum(schematic_count(fields[-1].lower())for fields in definitions[name]if fields[0].lower().startswith('x'))
    transistors=sum(fields[0].lower().startswith('m')for fields in definitions[active])
    if transistors!=6:raise ValueError('the active SRAM bitcell does not contain six MOS devices')
    lef=(root/(native+'.lef')).read_text()
    if not re.search(r'\bMACRO\s+'+re.escape(native)+r'\b',lef):raise ValueError('native macro missing from LEF')
    lef_ports=set(re.findall(r'^\s*PIN\s+(\S+)',lef,re.M))
    schematic_ports=set(ports[native.lower()])
    if lef_ports!=schematic_ports:raise ValueError('LEF/schematic pin sets differ: '+str(sorted(lef_ports^schematic_ports)))
    labels={}
    for layer in layout.layer_indexes():
        if layout.get_info(layer).datatype!=251:continue
        for shape in top.shapes(layer).each():
            if shape.is_text():
                text=shape.text;point=text.trans.disp
                labels.setdefault(text.string,[]).append((point.x*layout.dbu,point.y*layout.dbu))
    for name in lef_ports:
        body=re.search(r'\bPIN\s+'+re.escape(name)+r'\s+(.*?)\bEND\s+'+re.escape(name)+r'(?=\s|$)',lef,re.S)
        rectangles=[list(map(float,m))for m in re.findall(r'\bRECT\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*;',body[1])]
        if not any(x0-1e-6<=x<=x1+1e-6 and y0-1e-6<=y<=y1+1e-6
                   for x,y in labels.get(name,[])for x0,y0,x1,y1 in rectangles):
            raise ValueError('LEF pin does not match a GDS port label: '+name)
    lib=(root/(native+'.lib')).read_text()
    if not re.search(r'\bcell\s*\(\s*'+re.escape(native)+r'\s*\)',lib):raise ValueError('native macro missing from Liberty')
    box=top.bbox()
    size=re.search(r'\bSIZE\s+([\d.]+)\s+BY\s+([\d.]+)\s*;',lef)
    if not size or abs(float(size[1])-box.width()*layout.dbu)>1e-6 or abs(float(size[2])-box.height()*layout.dbu)>1e-6:
        raise ValueError('LEF does not enclose the same full-geometry footprint as GDS')
    result=dict(native_macro=native,gds_bitcells=gds_count(top.cell_index()),
        schematic_bitcells=schematic_count(native.lower()),bitcell_transistors=transistors,
        gds_cells=layout.cells(),gds_dbu_um=layout.dbu,width_um=box.width()*layout.dbu,
        height_um=box.height()*layout.dbu,pins=len(lef_ports),pin_sets_match=True,gds_pin_labels_match_lef=True,
        native_cell_names_match=True,full_geometry_footprint_matches=True,drc_passed=False,lvs_passed=False,
        note='Hierarchy/bitcell/pin-name checks only; no routed transistor connectivity or design-rule verification.')
    (root/'inspection.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(inspect(sys.argv[1],sys.argv[2])))
