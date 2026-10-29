"""Bind synthesis area to the exact simulated hardware, independently of timing."""
import hashlib
import json
import math
from pathlib import Path


def load_area(path):
    path=Path(path).resolve();proof=json.loads(path.read_text())
    if proof.get('evidence_kind')!='standard_cell_synthesis':raise ValueError('expected standard-cell synthesis evidence')
    net=Path(proof['netlist']);statistics=net.parent/'cell_statistics.json'
    if hashlib.sha256(net.read_bytes()).hexdigest()!=proof.get('mapped_sha256'):
        raise ValueError('area netlist changed')
    library=Path(proof.get('liberty_file','missing'))
    if hashlib.sha256(library.read_bytes()).hexdigest()!=proof.get('liberty_sha256'):
        raise ValueError('area process library changed')
    actual=json.loads(statistics.read_text())['design']['area']
    if not math.isfinite(actual)or actual<=0 or not math.isclose(actual,proof['cell_area_um2'],rel_tol=1e-12):
        raise ValueError('area report disagrees with synthesis cell statistics')
    inventory=net.parent/'memory_inventory.json'
    if inventory.exists():
        memory=json.loads(inventory.read_text())
        if memory['netlist_sha256']!=proof['mapped_sha256']:raise ValueError('memory inventory belongs to another netlist')
        proof['memory_inventory']=memory
    return dict(proof=proof,path=str(path),report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                statistics_sha256=hashlib.sha256(statistics.read_bytes()).hexdigest())


def area_metrics(metrics,evidence,scope):
    proof=evidence['proof']
    if proof.get('scope')!=scope or any(proof.get(key)!=metrics.get(key)for key in('stages','rows','cols','rtl_sha256')):
        raise ValueError('area evidence does not cover this exact simulated hardware scope')
    area=proof['cell_area_um2']
    result=dict(cell_area_um2=area,area_scope={'array':'array_core','gemm':'gemm_system','system':'tile_system'}.get(scope,scope),
        area_kind=proof['evidence_kind'],area_netlist_sha256=proof['mapped_sha256'],
        area_report=evidence['path'],area_report_sha256=evidence['report_sha256'],
        area_statistics_sha256=evidence['statistics_sha256'],area_liberty_sha256=proof['liberty_sha256'],
        area_mapping_profile=proof.get('mapping_profile','unspecified'),
        area_timing_certified=False)
    if 'memory_inventory'in proof:
        result['area_retained_memory_bytes']=proof['memory_inventory']['retained_bytes']
        result['area_full_storage_capacity']=proof['memory_inventory']['all_nominal_bits_retained']
    gops=metrics.get('gops',2*metrics['gmac_s']if'gmac_s'in metrics else None)
    if gops is not None:result['gops_per_mm2']=gops/(area/1e6)
    return result
