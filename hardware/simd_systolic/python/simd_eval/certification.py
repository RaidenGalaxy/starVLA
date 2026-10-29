"""Validate a core timing claim against extracted and routed numerical evidence."""
import hashlib
import math
from pathlib import Path
from .constraints import core_policy
from .architecture import ARCH


def core_certificate(proof,numeric,maximum_period_ns=1):
    failures=[]
    scope=proof.get('scope')
    if proof.get('stages')!=2 or proof.get('process')!='asap7' or proof.get('architecture_sha256')!=ARCH['architecture_sha256']:
        failures.append('proof is not the current ASAP7 F2 architecture')
    if scope not in('pe','array','fp64'):failures.append('unsupported core scope')
    for key in('period_ns','setup_wns_ns','hold_wns_ns','tns_ns'):
        value=proof.get(key)
        if not isinstance(value,(int,float))or not math.isfinite(value):
            failures.append('missing/nonfinite '+key)
    if failures:return dict(passed=False,failures=failures)
    if not 0<proof['period_ns']<=maximum_period_ns:failures.append('clock period exceeds target')
    if proof['setup_wns_ns']<0 or proof['hold_wns_ns']<0 or proof['tns_ns']!=0:
        failures.append('setup/hold/TNS did not close')
    for key in('setup_hold_closed','electrical_clean','coverage_clean','route_drc_clean','constraint_policy_clean'):
        if proof.get(key)is not True:failures.append('incomplete '+key)
    if proof.get('timing_status')not in('array_closed','kernel_closed'):failures.append('timing is not certified')
    if proof.get('provenance_status','').startswith('historical'):failures.append('historical provenance')
    required=('netlist','sdc','spef','setup','hold','electrical','coverage','coverage_analysis','drc')
    for name in set(required)|set(proof.get('evidence',{})):
        path=Path(proof.get('evidence',{}).get(name,''))
        if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=proof.get('evidence_sha256',{}).get(name):
            failures.append('missing/changed '+name)
    # Early Nangate45 records stored the library digest but no absolute path.
    # Authenticate those against this project's pinned library, never an
    # arbitrary installed library or a digest inferred from its filename.
    library=Path(proof.get('liberty_file','missing'))
    if not library.is_file()or hashlib.sha256(library.read_bytes()).hexdigest()!=proof.get('liberty_sha256'):
        failures.append('missing/changed process library')
    net=Path(proof.get('evidence',{}).get('netlist',''))
    flow=net.parent/'flow.tcl'
    if not flow.is_file()or hashlib.sha256(flow.read_bytes()).hexdigest()!=proof.get('flow_sha256'):
        failures.append('missing/changed implementation flow')
    sdc=Path(proof.get('evidence',{}).get('sdc',''))
    if sdc.is_file()and not (core_policy(sdc.read_text(),proof['period_ns'])['clean'] and core_policy(sdc.read_text(),proof['period_ns'])['explicit_units']):
        failures.append('clock/IO/exception policy')
    for field,evidence in [('netlist_sha256','netlist'),('sdc_sha256','sdc')]:
        if not proof.get(field)or proof[field]!=proof.get('evidence_sha256',{}).get(evidence):
            failures.append('inconsistent '+field)
    if not(numeric.get('passed')is True and numeric.get('mapped_netlist_kind')=='routed'):
        failures.append('routed numerical verification missing')
    for key in('scope','stages','rtl_sha256','netlist_sha256'):
        if proof.get(key)is None or numeric.get(key)!=proof[key]:failures.append('numerical scope mismatch: '+key)
    if scope in('pe','array'):
        if numeric.get('all_formats')is not True:failures.append('not all formats numerically checked')
        if numeric.get('liberty_sha256')!=proof.get('liberty_sha256'):failures.append('numerical process library differs')
    if scope=='array':
        for key in('wave_drain_protocol_checked','all_wave_stages_tracked'):
            if numeric.get(key)is not True:failures.append('missing safe wave drain evidence: '+key)
        for key in('rows','cols'):
            if proof.get(key)is None or numeric.get(key)!=proof[key]:failures.append('numerical dimensions differ: '+key)
    return dict(passed=not failures,failures=failures)


def compatible_core_support(pe,array):
    """Both physical scopes must use the same F2 arithmetic support."""
    if pe.get('stages')!=2 or array.get('stages')!=2:return False
    if pe.get('architecture_sha256')!=ARCH['architecture_sha256'] or array.get('architecture_sha256')!=ARCH['architecture_sha256']:return False
    names=('ControlReplicas.sv','PrefixAdd.sv','fpnew_pkg.sv','fpnew_fma.sv',
           'SimdFpF2.sv','SimdFpF2Core.sv')
    left,right=pe.get('input_manifest',{}),array.get('input_manifest',{})
    return all(left.get(name)is not None and left[name]==right.get(name)for name in names)

# Compatibility for historical report readers; new evidence is always F2.
compatible_hf_support=compatible_core_support
