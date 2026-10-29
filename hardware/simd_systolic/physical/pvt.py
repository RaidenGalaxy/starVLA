"""Compare the actual Liberty operating points, not just corner labels."""
import hashlib
import math
import re
from pathlib import Path


def operating_point(path):
    path=Path(path);raw=path.read_text();point={};problems=[]
    selected=re.search(r'\bdefault_operating_conditions\s*:\s*"?([^;"\s]+)"?\s*;',raw)
    condition=None
    if selected:
        condition=re.search(r'\boperating_conditions\s*\(\s*"?'+re.escape(selected[1])+r'"?\s*\)\s*\{([^{}]*)\}',raw,re.S)
        if not condition:problems.append('default operating condition is missing')
    for name,active_name in [('nom_process','process'),('nom_voltage','voltage'),('nom_temperature','temperature')]:
        source=condition[1]if condition else raw
        attribute=active_name if selected else name
        match=re.search(r'\b'+attribute+r'\s*:\s*([-+\d.eE]+)\s*;',source)
        if not match:
            problems.append('missing '+name);point[name]=None;continue
        value=float(match[1])
        if not math.isfinite(value):problems.append('invalid '+name)
        point[name]=value
    return dict(file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        values=point,default_operating_condition=selected[1]if selected else None,problems=problems)


def matching_points(standard_cells,macro):
    standard=operating_point(standard_cells);memory=operating_point(macro)
    problems=['standard cells: '+x for x in standard['problems']]+['SRAM: '+x for x in memory['problems']]
    for name,value in standard['values'].items():
        other=memory['values'][name]
        if value is not None and other is not None and abs(value-other)>1e-12:
            problems.append(name+' differs between standard cells and SRAM')
    return dict(matched=not problems,standard_cells=standard,macro=memory,problems=problems)
