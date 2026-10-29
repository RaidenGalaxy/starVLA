#!/usr/bin/env python3
"""Collect small, reviewable evidence without treating stale reports as current."""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from simd_eval.area import load_area
from simd_eval.certification import core_certificate,compatible_hf_support


def collect(output):
    output.mkdir(parents=True,exist_ok=True)
    # Remove only this collector's generated snapshots, including numbered
    # names from older runs. Stable source identities prevent stale duplicates.
    for snapshot in output.glob('*.json'):
        if re.fullmatch(r'(?:\d{3}|[0-9a-f]{12})_.+\.json',snapshot.name):snapshot.unlink()
    builds=[json.loads(p.read_text())for p in (ROOT/'.build').glob('*/capabilities.json')
            if not(p.parent/'.incomplete').exists()]
    records=[];incomplete=[]
    candidates=list((ROOT/'.build').glob('*/verification*.json'))
    candidates+=list((ROOT/'.build').glob('auxiliary_*/verification.json'))
    candidates+=list((ROOT/'artifacts').rglob('metrics.json'))
    candidates+=list((ROOT/'artifacts').rglob('*comparison*.json'))
    candidates+=list((ROOT/'artifacts/workloads').rglob('*.json'))
    candidates+=list((ROOT/'artifacts').rglob('verification*.json'))
    candidates+=list((ROOT/'artifacts').rglob('figure.json'))
    candidates+=list((ROOT/'artifacts').rglob('*crosscheck.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/timing.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/cts_timing.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/synthesis.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/frequency_sweep.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/gate_verification.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/verification*.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/failure.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/interrupted.json'))
    candidates+=list((ROOT/'.build/physical').glob('*/memory_inventory.json'))
    for index,path in enumerate(sorted(set(candidates))):
        relative=str(path.relative_to(ROOT));raw=path.read_bytes()
        try:data=json.loads(raw)
        except json.JSONDecodeError:
            incomplete.append(relative);continue
        if 'artifacts/workloads/' in relative and not('npz_sha256'in data or'captured_phases'in data):continue
        current=None
        if data.get('rtl_sha256'):
            current=any(all(data.get(key)==cap.get(key)for key in
                ('rtl_sha256',)+tuple(k for k in('simulator_sha256','testbench_sha256','dram_library_sha256')if k in data))
                for cap in builds)
        source_id=hashlib.sha256(relative.encode()).hexdigest()[:12]
        name=f'{source_id}_{path.parent.name}_{path.name}'
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
        records.append({'source':relative,'snapshot':name,'sha256':hashlib.sha256(raw).hexdigest(),
          'matches_current_build':current,'matches_available_frozen_build':current,'report':data})
    summary={'schema_version':1,'generated_utc':datetime.now(timezone.utc).isoformat(),
      'acceptance':{'full_plan_complete':False,'one_ghz_upgradeability_certified':False,
        'full_array_one_ghz_certified':False,
        'reason':'Missing closed single-PE/2x2/8x8 extracted timing, full-system memory/CDC signoff and complete workload suite; see STATUS.md'},
      'reports':records,'incomplete_or_invalid_json_reports':incomplete}
    fp64_proofs=[];core_proofs=[]
    for record in records:
        proof=record['report']
        if proof.get('scope')not in('fp64','pe','array')or proof.get('timing_status')not in('kernel_closed','array_closed'):continue
        directory=(ROOT/record['source']).parent
        gates=directory/'gate_verification_routed.json'
        if not gates.exists():gates=directory/'gate_verification.json'
        if not gates.exists():continue
        numeric=json.loads(gates.read_text())
        if not core_certificate(proof,numeric)['passed']:continue
        core_proofs.append(record)
        if proof['scope']=='fp64':fp64_proofs.append(record['source'])
    pe=[r for r in core_proofs if r['report']['scope']=='pe']
    array2=[r for r in core_proofs if r['report']['scope']=='array'and(r['report'].get('rows'),r['report'].get('cols'))==(2,2)]
    array8=[r for r in core_proofs if r['report']['scope']=='array'and(r['report'].get('rows'),r['report'].get('cols'))==(8,8)]
    pairs=[dict(pe=p['source'],array2=a['source'])for p in pe for a in array2 if compatible_hf_support(p['report'],a['report'])]
    summary['acceptance'].update(one_ghz_upgradeability_certified=bool(pairs),full_array_one_ghz_certified=bool(array8))
    remaining=['Full-system memory/CDC physical signoff and final complete-suite acceptance remain outstanding; see STATUS.md']
    if not pairs:remaining.append('No compatible HF single-PE/2x2 pair has closed extracted 1ns timing and all-format routed numerical verification')
    if not array8:remaining.append('No complete 8x8 core has closed extracted 1ns timing and all-format routed numerical verification')
    summary['acceptance']['reason']='; '.join(remaining)
    summary['milestones']={'fp64_arithmetic_one_ghz_certified':bool(fp64_proofs),
        'fp64_arithmetic_evidence':fp64_proofs,'scope_note':'FP64 arithmetic only; full PE and array require their own closed timing evidence.'}
    summary['milestones'].update(one_ghz_pe_array2_pairs=pairs,full_array_one_ghz_evidence=[r['source']for r in array8])
    full_storage=[]
    for record in records:
        report=record['report']
        if report.get('scope')!='gemm'or not report.get('memory_inventory',{}).get('all_nominal_bits_retained'):continue
        if report.get('evidence_kind')!='standard_cell_synthesis':continue
        evidence=load_area(ROOT/record['source'])
        if evidence['proof'].get('memory_inventory',{}).get('all_nominal_bits_retained'):
            full_storage.append(record['source'])
    summary['milestones'].update(full_storage_synthesis_area_verified=bool(full_storage),
        full_storage_area_evidence=full_storage)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 已执行的验证与评估记录','',
      f'全部计划尚未验收；完整 PE/子阵列 1GHz 提频能力：{bool(pairs)}；完整 8×8 核心 1GHz：{bool(array8)}。性能中的配置时钟预测不等于芯片实测。',
      f'FP64 算术核心的 1GHz 提取后时序与布线网表数值验证通过：{bool(fp64_proofs)}。该证据不覆盖完整 PE/阵列。',
      '', '构建匹配指对应的冻结构建仍可校验，不表示报告属于最新源码。',
      '', '| 来源 | 证据 | 保留构建匹配 |','|---|---|---|']
    for r in records:
        d=r['report']
        evidence=(f"captured phases={d['captured_phases']}"if'captured_phases'in d else
          f"precision comparisons={list(d)}"if d and all(isinstance(v,dict)and'speedup'in v for v in d.values())else
          f"{d.get('phase','')} {d.get('evaluated_shape_mnk','')}; captured={d.get('source','')}"if'npz_sha256'in d and'phase'in d else
          f"{d.get('baseline_operand_mode',d.get('baseline_format'))} → {d.get('candidate_operand_mode',d.get('candidate_format'))}; speedup={d['speedup']:.6f}; {d.get('time_kind')}"if'speedup'in d else
          f"{d.get('vectors',d.get('reads',''))} vectors/reads; passed={d.get('passed')}"if'passed'in d else
          f"{d.get('format','')} {d.get('m','')}×{d.get('n','')}×{d.get('k','')}; exact={d.get('exact_output_match',d.get('exact_hardware_reference_match'))}"if'format'in d else
          f"{d.get('timing_status',d.get('status','frequency sweep'))}; WNS={d.get('setup_wns_ns','')} ns")
        lines.append(f"| `{r['source']}` | [{evidence}]({r['snapshot']}) | {r['matches_current_build']} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(f'Collected {len(records)} evidence records into {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results');collect(p.parse_args().out)
