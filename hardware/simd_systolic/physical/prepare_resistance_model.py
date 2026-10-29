#!/usr/bin/env python3
"""Build a separate OSDI model using equivalent voltage-form resistor stamps.

ASAP7 has zero external source/drain sheet resistance; CMG107 clamps it to
1 milliohm. Stamping a 1000S conductance beside tiny off-state conductances
can make the matrix ill-conditioned. V=R*I retains exactly the same resistance
and constitutive relation as I=V/R. The original OSDI is never overwritten.
"""
import json
import re
import subprocess
from pathlib import Path

from prepare_analog import BASE,prepare,sha


def voltage_resistance_port(text):
    for terminal,resistance in [('d','Rdrain'),('s','Rsource')]:
        node=terminal+'i'
        pattern=rf'I\({terminal},\s*{node}\)\s*<\+\s*V\({terminal},\s*{node}\)\s*/\s*{resistance}\s*;'
        text,count=re.subn(pattern,f'V({terminal}, {node}) <+ I({terminal}, {node}) * {resistance};',text)
        if count!=1:raise ValueError('unexpected upstream source/drain resistor stamp')
    return text


def prepare_voltage_resistance():
    original=prepare();source=Path(original['osdi']).parent
    out=BASE/'bsimcmg107_voltage_resistance';out.mkdir(exist_ok=True);inputs={}
    for name in original['port_sources_sha256']:
        text=(source/name).read_text()
        if name=='bsimcmg_body.include':text=voltage_resistance_port(text)
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
        'resistance_formulation':'voltage_branch_V_equals_R_times_I',
        'resistance_parameters_unchanged':True,'upstream_resistor_expression_rewritten':True,
        'compiler_port':original['compiler_port']+'; equivalent voltage-branch source/drain resistors',
        'signoff_toolchain':False}
    manifest.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(prepare_voltage_resistance(),indent=2))
