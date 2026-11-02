#!/usr/bin/env python3
"""Copy an immutable corrected system checkpoint with its separate actual STA.

No EDA mutation is performed; this joins two authenticated diagnostic parents
so subsequent repairs can select actual paths without altering either parent.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
from run import ROOT,sha
from async_mapping_policy import validate_checkpoint


def run(args):
    source=args.source.resolve();sta=args.sta.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('preserve prior joined checkpoints')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'attachment_driver.py')
    state=dict(passed=False,active_phase='wait_for_actual_sta',physical_frequency_certified=False)
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        deadline=time.monotonic()+args.wait_seconds
        while not(sta/'verification.json').exists():
            if time.monotonic()>=deadline:raise TimeoutError('actual system STA did not finish')
            time.sleep(5)
        prior=json.loads((source/'verification.json').read_text());timed=json.loads((sta/'verification.json').read_text())
        validate_checkpoint(source,prior)
        if(not timed.get('passed')or not timed.get('reused_immutable_checkpoint_only')or
                timed['source_verification_sha256']!=sha(source/'verification.json')or
                (timed['core_period_ns'],timed['memory_period_ns'])!=(1,2)or
                set(timed['timing'])!={'setup_wns_ns','hold_wns_ns'}):
            raise ValueError('STA does not bind the corrected immutable checkpoint')
        for record in(prior,timed):
            for filename,digest in record['evidence'].items():
                if sha(filename)!=digest:raise ValueError('parent evidence changed: '+filename)
        state['active_phase']='copy_authenticated_checkpoint';save()
        names=('repaired.odb','repaired.v','cdc_constraints.tcl','protected_power.tcl','reset_repair.tcl')
        for name in names:
            shutil.copy2(source/name,out/name)
            if sha(source/name)!=sha(out/name):raise ValueError('checkpoint copy differs: '+name)
        for name in('setup.rpt','clocks.rpt','openroad.log'):
            shutil.copy2(sta/name,out/name)
            if sha(sta/name)!=sha(out/name):raise ValueError('actual STA report copy differs')
        (out/'design.sdc').write_text((sta/'design.sdc').read_text().replace(str(sta),str(out)))
        record=dict(kind='immutable_corrected_checkpoint_with_actual_separate_sta',passed=True,
            source_verification=str(source/'verification.json'),source_verification_sha256=sha(source/'verification.json'),
            source_sta_verification=str(sta/'verification.json'),source_sta_verification_sha256=sha(sta/'verification.json'),
            source_rtl_sha256=prior['source_rtl_sha256'],source_mapped_sha256=prior['source_mapped_sha256'],
            physical_macro_package_manifest_sha256=prior['physical_macro_package_manifest_sha256'],
            core_period_ns=1,memory_period_ns=2,timing=timed['timing'],
            logical_roots_and_protected_data_clock_masters_preserved=True,
            no_eda_mutation_performed=True,source_sdc_semantics_preserved=True,
            physical_frequency_certified=False,extracted_sta=False,
            **{k:prior[k]for k in('qualified_async_mapping_root','async_mapping_correction_verified',
                'async_correction_verification','async_correction_verification_sha256')if k in prior},
            evidence={str(p):sha(p)for p in [source/'verification.json',sta/'verification.json',
                *[Path(p)for p in timed['evidence']if p.endswith('.lib')],
                *[p for p in out.iterdir()if p.is_file()and p.name!='state.json']]})
        (out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
        state.update(passed=True,timing=timed['timing'])
    except Exception as error:
        state['error']=str(error);raise
    finally:
        state.pop('active_phase',None);save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in('source','sta','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=1800);a=p.parse_args()
    if not 0<=a.wait_seconds<=3600:p.error('wait0..3600seconds')
    run(a)
