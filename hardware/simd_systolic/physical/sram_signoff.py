"""Production SRAM evidence audit at the locked standard-cell PVT points.

The frozen legacy auditor supplies GDS/netlist/report integrity checks. Its
fixed temperature table is superseded here by actual selected Liberty points;
each characterization record is independently reauthenticated before use.
"""
import hashlib
import argparse
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from audit_sram import audit as legacy_audit,identities_without_evidence
from asap7_platform import profile
from pvt import matching_points


def audit(root=None):
    root=Path(root or Path(__file__).resolve().parents[1]/'.deps/asap7_sram').resolve()
    result=legacy_audit(root);macro=result['macro'];path=root/'physical_evidence.json'
    evidence=json.loads(path.read_text())if path.exists()else{}
    expected_views=identities_without_evidence(result['views_sha256'])
    gaps=[g for g in result['gaps']if not g.startswith('Missing authenticated characterization_')]
    corners={};checks={}
    for corner in ('TT','SS','FF'):
        record=evidence.get('characterization_'+corner,{})
        library=(root/record.get('liberty_file','missing')).resolve()
        report=(root/record.get('file','missing')).resolve()
        inside=(root in library.parents and root in report.parents)
        valid=(inside and library.is_file()and report.is_file()and record.get('passed')is True and
            record.get('verified_period_ps')==result['memory_period_ps']and evidence.get('macro')==macro and
            evidence.get('views_sha256')==expected_views and
            hashlib.sha256(report.read_bytes()).hexdigest()==record.get('sha256'))
        comparison=matching_points(profile(corner)['liberty'],library)if inside and library.is_file()else None
        if valid:
            valid=bool(comparison['matched']and re.search(r'\bcell\s*\(\s*"?'+re.escape(macro)+r'"?\s*\)',library.read_text()))
        checks[corner]=dict(authenticated=valid,pvt_comparison=comparison)
        if valid:corners[corner]=str(library)
        else:gaps.append('Missing authenticated characterization_'+corner+' at the locked standard-cell PVT point')
    result.update(gaps=gaps,physical_certified=not gaps,functional_model_only=bool(gaps),
        corner_libraries=corners,corner_validation=checks,
        audit_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pvt_policy='actual default operating conditions, with nominal fallback, of locked ASAP7 standard-cell libraries')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/migration_asap7_f2/production_sram_audit.json')
    parser.add_argument('--require-certified',action='store_true');args=parser.parse_args()
    report=audit(args.root);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ('macro','physical_certified','gaps')},indent=2))
    if args.require_certified and not report['physical_certified']:raise SystemExit(1)
