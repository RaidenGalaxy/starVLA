#!/usr/bin/env python3
"""Separate diagnostic model collapsing nominally zero sheet resistances.

Only when RDSMOD=RGEOMOD=RSHS/RSHD=0 does this remove CMG107's numerical
1 milliohm floor and collapse the corresponding external/internal nodes.
Intrinsic RDSW and every ASAP7 parameter remain unchanged. This differs from
the canonical model and cannot be used as unmodified-model signoff evidence.
"""
import json
import re
import subprocess
from pathlib import Path

from prepare_analog import BASE,prepare,sha


def zero_sheet_port(text):
    for terminal,resistance,sheet in [('d','Rdrain','RSHD'),('s','Rsource','RSHS')]:
        node=terminal+'i'
        pattern=rf'I\({terminal},\s*{node}\)\s*<\+\s*V\({terminal},\s*{node}\)\s*/\s*{resistance}\s*;'
        replacement=(f'if(RDSMOD==0 && RGEOMOD==0 && {sheet}==0) V({terminal}, {node}) <+ 0;\n'
                     f'else I({terminal}, {node}) <+ V({terminal}, {node}) / {resistance};')
        text,count=re.subn(pattern,replacement,text)
        if count!=1:raise ValueError('unexpected upstream source/drain resistor stamp')
    return text


def prepare_zero_sheet():
    original=prepare();source=Path(original['osdi']).parent
    out=BASE/'bsimcmg107_zero_sheet';out.mkdir(exist_ok=True);inputs={}
    for name in original['port_sources_sha256']:
        text=(source/name).read_text()
        if name=='bsimcmg_body.include':text=zero_sheet_port(text)
        (out/name).write_text(text);inputs[name]=sha(out/name)
    output=out/'bsimcmg.osdi';manifest=out/'capabilities.json'
    prior=json.loads(manifest.read_text())if manifest.is_file()else{}
    if not(output.is_file()and prior.get('port_sources_sha256')==inputs and
            prior.get('osdi_sha256')==sha(output)and prior.get('compiler_sha256')==original['compiler_sha256']):
        with(out/'compile.log').open('w')as stream:
            subprocess.run([str(BASE/'openvaf/openvaf'),str(out/'bsimcmg.va'),'-o',str(output)],
                stdout=stream,stderr=subprocess.STDOUT,check=True)
    result={**original,'osdi':str(output),'osdi_sha256':sha(output),'port_sources_sha256':inputs,
        'preparation_driver_sha256':sha(__file__),'parent_osdi_sha256':original['osdi_sha256'],
        'resistance_formulation':'zero_sheet_topology_collapse','numerical_floor_removed_ohm':.001,
        'asap7_parameters_unchanged':True,'intrinsic_rdsw_unchanged':True,'equations_modified':True,
        'compiler_port':'Separate diagnostic model: collapse only nominally zero external sheet resistances',
        'signoff_toolchain':False}
    manifest.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(prepare_zero_sheet(),indent=2))
