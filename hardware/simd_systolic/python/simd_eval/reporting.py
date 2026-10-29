"""Evidence identities and strict comparisons of hardware experiments."""
import hashlib
import json
import math
from fractions import Fraction
from decimal import Decimal,localcontext
from pathlib import Path


def workload_hash(a, b):
    # Hash numerical values, not container metadata or NPZ compression timestamps.
    from fractions import Fraction
    payload = [[[str(Fraction(x)) for x in row] for row in matrix] for matrix in (a, b)]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def numerical_errors(bits, reference, fmt, scale=1, zero_point=0):
    """Accumulate exact squared errors before conversion to report numbers."""
    from .reference import decode,Format,WIDTHS,signed
    if not scale:raise ValueError('output reconstruction requires a nonzero scale')
    squared=norm=Fraction(0);count=nonfinite=0
    for row,expected in zip(bits,reference):
        for encoded,target in zip(row,expected):
            count+=1
            if Format(fmt)<Format.F4:actual=Fraction(signed(encoded,WIDTHS[Format(fmt)]))
            else:
                value=decode(encoded,fmt)
                if value.kind!='finite':nonfinite+=1;continue
                actual=(-1 if value.sign else 1)*value.magnitude
            actual=(actual-zero_point)/scale
            squared+=(actual-target)**2;norm+=target**2
    if nonfinite:
        return dict(rmse=None,relative_l2=None,nonfinite_output_count=nonfinite,error_metric_status='nonfinite_hardware_output')
    with localcontext()as context:
        context.prec=40
        decimal=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
        rmse=decimal(squared/count).sqrt()
        relative=decimal(squared/norm).sqrt()if norm else Decimal(0)if not squared else Decimal('Infinity')
        def finite(x):
            value=float(x)
            return value if math.isfinite(value)and(value!=0 or x==0)else None
        r,l=finite(rmse),finite(relative)
        return dict(rmse=r,relative_l2=l,rmse_decimal=str(rmse),relative_l2_decimal=str(relative),
                    nonfinite_output_count=0,error_metric_status='finite'if r is not None and l is not None else'outside_float_report_range')


def write_metrics(out, metrics):
    import csv
    import io
    import tempfile
    out = Path(out)
    metrics.update(runtime_clock_metrics(metrics))
    stream=io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(metrics))
    writer.writeheader();writer.writerow(metrics)
    texts={'metrics.json':json.dumps(metrics,indent=2,allow_nan=False)+'\n','metrics.csv':stream.getvalue()}
    for name,text in texts.items():
        with tempfile.NamedTemporaryFile(mode='w',dir=out,prefix='.'+name+'.',delete=False,newline='')as temporary:
            temporary.write(text);pending=Path(temporary.name)
        pending.replace(out/name)


def dram_geometry(stdout):
    records=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('P ')]
    geometry=dict(dram_burst_bytes=64,dram_geometry_observed=False)
    burst=64
    if records:
        if len(records)!=1 or len(records[0])!=3:raise ValueError('ambiguous DRAM geometry/submission evidence')
        burst,reads,writes=records[0]
        if burst<8 or burst>4096 or burst&(burst-1)or min(reads,writes)<0:raise ValueError('invalid DRAM geometry')
        geometry.update(dram_burst_bytes=burst,dram_submitted_read_transactions=reads,dram_submitted_write_transactions=writes,dram_geometry_observed=True)
    clocks=[tuple(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('K ')]
    geometry['dram_runtime_geometry_observed']=bool(clocks)
    if clocks:
        if any(len(x)!=3 or x!=clocks[0]for x in clocks):raise ValueError('ambiguous runtime DRAM clocks')
        tck,bits,bl=clocks[0]
        if min(tck,bits,bl)<=0 or bits*bl!=8*burst:raise ValueError('DRAM clock/geometry mismatch')
        geometry.update(dram_tck_ps=tck,dram_frequency_hz=1e12/tck,
                        dram_data_rate_MT_s=2e6/tck,dram_bus_bits=bits,dram_burst_length=bl,
                        dram_peak_shared_bus_GB_s=2*bits/8*1000/tck,
                        dram_clock_source='DRAMSim3 runtime GetTCK/GetBusBits/GetBurstLength')
    ports=[tuple(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('PORTS ')]
    geometry['runtime_port_monitor_observed']=bool(ports)
    if ports:
        if len(ports)!=1 or len(ports[0])!=4 or min(ports[0])<0:raise ValueError('invalid observed port monitor')
        mr,mw,rb,wb=ports[0]
        geometry.update(dma_read_max_inflight_observed=mr,dma_write_max_inflight_observed=mw,
                        dma_read_beats_observed=rb,dma_write_beats_observed=wb,
                        dma_read_observed_bus_bytes=16*rb,dma_write_observed_bus_bytes=16*wb)
    return geometry


def runtime_clock_metrics(metrics):
    """Do not replace historical periods or infer physical frequency closure."""
    result={}
    for domain in ('core','memory'):
        f=metrics.get(domain+'_frequency_hz')
        if f is not None:
            if not math.isfinite(f)or f<=0:raise ValueError('invalid runtime clock')
            result[domain+'_period_ps']=1e12/f
    if 'memory_frequency_hz'in metrics:
        result.update(writeback_clock_domain='memory',
                      dma_read_peak_GB_s=16*metrics['memory_frequency_hz']/1e9,
                      dma_write_peak_GB_s=16*metrics['memory_frequency_hz']/1e9,
                      dma_peak_is_per_direction=True,full_duplex_throughput_guaranteed=False)
    if 'elapsed_ps'in metrics and 'submitted_ps'in metrics:
        elapsed,submitted=metrics['elapsed_ps'],metrics['submitted_ps']
        if not 0<=submitted<=elapsed:raise ValueError('posted completion exceeds final drain time')
        result.update(command_to_posted_completion_ps=submitted,command_to_final_drain_ps=elapsed,
                      final_drain_tail_ps=elapsed-submitted)
    return result


def dma_link_metrics(metrics,read_requests,write_requests):
    """Account for independent 128-bit read/write data ports and 64B requests."""
    seconds=metrics['elapsed_ps']*1e-12;frequency=metrics['memory_frequency_hz']
    if seconds<=0 or frequency<=0 or min(read_requests,write_requests)<0:raise ValueError('invalid DMA link observations')
    read=64*read_requests;write=64*write_requests;peak=16*frequency
    if max(read,write)>seconds*peak+16:raise ValueError('DMA traffic exceeds a physical data port capacity')
    return dict(dma_transfer_bytes=read+write,dma_read_bus_bytes=read,dma_write_bus_bytes=write,
        dma_read_bandwidth_GB_s=read/seconds/1e9,dma_write_bandwidth_GB_s=write/seconds/1e9,
        dma_bus_bandwidth_GB_s=(read+write)/seconds/1e9,dma_peak_per_direction_GB_s=peak/1e9,
        dma_read_bus_utilization=read/(seconds*peak),dma_write_bus_utilization=write/(seconds*peak),
        dma_link_definition='independent 128-bit read and write data ports; four beats per logical 64B request; includes byte-disabled lanes')


def dram_evidence(stdout,logical_reads,logical_writes,physical_reads,physical_writes,enabled=True):
    geometry=dram_geometry(stdout)
    if geometry.get('runtime_port_monitor_observed')and(geometry['dma_read_beats_observed'],geometry['dma_write_beats_observed'])!=(4*logical_reads,4*logical_writes):
        raise ValueError('observed interface data beats differ from requests')
    if enabled:
        if 'dram_submitted_read_transactions'in geometry:
            fragments=max(1,64//geometry['dram_burst_bytes'])
            if(geometry['dram_submitted_read_transactions'],geometry['dram_submitted_write_transactions'])!=(logical_reads*fragments,logical_writes*fragments):
                raise ValueError('DRAM adapter fragments disagree with RTL requests')
            if physical_reads>logical_reads*fragments or physical_writes>logical_writes*fragments:
                raise ValueError('physical DRAM bursts exceed submissions')
        elif(logical_reads,logical_writes)!=(physical_reads,physical_writes):
            raise ValueError('RTL requests and DRAM physical bursts disagree')
    return {**geometry,'dram_physical_bytes':geometry['dram_burst_bytes']*(physical_reads+physical_writes)}


def memory_evidence(stdout, caps, read_bursts, write_bursts, read_bytes, write_bytes, dram=True):
    """Cross-check independent RTL counters and external transaction observations."""
    records=[list(map(int,line.split()[1:])) for line in stdout.splitlines() if line.startswith('B ')]
    if len(records)!=1 or len(records[0])!=7:
        raise ValueError('missing RTL DMA evidence')
    rb,wb,rd,wr,shadow,external_rb,external_wb=records[0]
    if ((rb,wb)!=(external_rb,external_wb) or (rd,wr,shadow)!=(read_bytes,write_bytes,write_bytes)):
        raise ValueError('RTL DMA, external memory and payload byte counts disagree')
    geometry=dram_evidence(stdout,rb,wb,read_bursts,write_bursts,dram)
    evidence={'dma_backend':caps.get('dma_backend','rtl'),
            'rtl_read_bursts':rb,'rtl_write_bursts':wb,
            'rtl_read_payload_bytes':rd,'rtl_write_payload_bytes':wr,
            'write_payload_byte_match':True,
            **geometry,**{key:caps[key] for key in ('simulator_sha256','testbench_sha256','dram_library_sha256')}}
    onchip=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('H ')]
    if caps.get('onchip_counters') and (len(onchip)!=1 or len(onchip[0])!=4):
        raise ValueError('missing on-chip traffic counters')
    if onchip:
        sr,sw,ar,aw=onchip[0]
        if sw!=read_bytes:raise ValueError('SRAM writes disagree with delivered DMA bytes')
        evidence.update(sram_read_bytes=sr,sram_write_bytes=sw,
                        aggregate_read_bytes=ar,aggregate_write_bytes=aw,
                        onchip_moved_bytes=sr+sw+ar+aw)
    overlap=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('U ')]
    if caps.get('dma_overlap_observation')and len(overlap)!=1:raise ValueError('missing DMA/core overlap observation')
    if overlap:
        if len(overlap)!=1 or len(overlap[0])!=2:raise ValueError('invalid DMA/core overlap observation')
        count,edges=overlap[0]
        if not 0<=count<=read_bytes or edges<0 or (count==0)!=(edges==0):raise ValueError('inconsistent DMA/core overlap byte count')
        evidence.update(dma_load_bytes_while_core_active=count,dma_load_edges_while_core_active=edges,
            overlap_observation='RTL DMA byte counter changes sampled at memory edges with coreActive high before the edge')
    return evidence


def utilization(vectors, feedback_stalls, input_stalls, cycles, macs, pes, lanes):
    """Separate issued PE vectors, active SIMD lanes and total MAC capacity."""
    capacity=cycles*pes
    if min(vectors,feedback_stalls,input_stalls,cycles,macs)>=0 and pes>0 and lanes>0:
        if vectors<=capacity and macs<=vectors*lanes and feedback_stalls+input_stalls<=cycles:
            return dict(issued_pe_vectors=vectors,feedback_stall_cycles=feedback_stalls,
                        input_wait_cycles=input_stalls,idle_pe_cycles=capacity-vectors,
                        array_issue_utilization=vectors/capacity if capacity else 0,
                        lane_utilization=macs/(vectors*lanes)if vectors else 0,
                        mac_utilization=macs/(capacity*lanes)if capacity else 0)
    raise ValueError('inconsistent RTL utilization counters')


def utilization_evidence(stdout,caps,cycles,macs,pes,lanes):
    records=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('V ')]
    if not records and not caps.get('utilization_counters'):
        return dict(mac_utilization=macs/(cycles*pes*lanes)if cycles else 0)
    if len(records)!=1 or len(records[0])!=3:raise ValueError('missing RTL utilization counters')
    return utilization(*records[0],cycles,macs,pes,lanes)


def sequencer_cycle_evidence(stdout,caps,jobs):
    names=caps.get('sequencer_cycle_categories',[])
    records=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('Z ')]
    if not names:
        if records:raise ValueError('unadvertised sequencer cycle counters')
        return {}
    seen=set();totals={name:0 for name in names}
    if len(totals)!=len(names):raise ValueError('duplicate sequencer cycle category')
    for record in records:
        tag,*values=record
        if tag in seen or tag not in jobs or len(values)!=len(names)or min(values)<0:
            raise ValueError('invalid sequencer cycle record')
        if sum(values)!=jobs[tag]['memory_cycles']:
            raise ValueError('sequencer categories do not sum to active memory cycles')
        seen.add(tag);jobs[tag]['sequencer_cycle_breakdown']=dict(zip(names,values))
        for name,value in zip(names,values):totals[name]+=value
    if seen!=set(jobs):raise ValueError('missing sequencer cycle records')
    return dict(sequencer_cycle_breakdown=totals,sequencer_memory_cycles=sum(totals.values()),
        sequencer_cycle_clock='memory',sequencer_cycle_accounting_exact=True,
        sequencer_cycle_scope='command active until completion; excludes idle/completion backpressure and final external DRAM drain')


def native_prefetch_evidence(stdout,caps,jobs):
    prefix=caps.get('native_prefetch_record_prefix','N')+' '
    records=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith(prefix)]
    if not caps.get('autonomous_prefetch'):
        if records:raise ValueError('unadvertised native prefetch counters')
        return {}
    names=('prefetched_tiles','prefetch_load_bytes','prefetch_load_bytes_during_tile_inflight','prefetch_load_bytes_while_core_active')
    totals=dict.fromkeys(names,0);seen=set()
    for record in records:
        if len(record)!=5:raise ValueError('invalid native prefetch counter record')
        tag,*values=record
        if tag in seen or tag not in jobs or min(values)<0 or values[2]>values[1]or values[3]>values[1]:
            raise ValueError('inconsistent native prefetch observations')
        seen.add(tag);jobs[tag].update(zip(names,values))
        for name,value in zip(names,values):totals[name]+=value
    if seen!=set(jobs):raise ValueError('missing native prefetch observations')
    return dict(**totals,native_prefetch_counter_clock='memory',
        native_prefetch_overlap_observer='testbench samples actual SRAM load handshake and coreActive before the memory edge; no functional CDC path')


def epilogue_evidence(stdout,caps,jobs,enabled=False):
    records=[list(map(int,line.split()[1:]))for line in stdout.splitlines()if line.startswith('Y ')]
    if not caps.get('output_epilogue'):
        if records or enabled:raise ValueError('unadvertised output conversion')
        return {}
    totals=dict.fromkeys(('epilogue_elements','epilogue_auxiliary_vectors','epilogue_memory_cycles','epilogue_saturations'),0)
    seen=set()
    for record in records:
        if len(record)!=5:raise ValueError('invalid output conversion counters')
        tag,*values=record
        if tag in seen or tag not in jobs or min(values)<0:raise ValueError('invalid output conversion tag/count')
        job=jobs[tag];expected=job['m']*job['n']if enabled else 0
        if values[0]!=expected or values[3]!=job.get('expected_output_saturations',0):
            raise ValueError('output element/saturation accounting mismatch')
        if enabled and not 0<values[1]<=2*expected:raise ValueError('output arithmetic operations omitted')
        if values[2]<values[1]or(not enabled and any(values)):raise ValueError('output conversion cycle accounting mismatch')
        seen.add(tag);job.update(zip(totals,values))
        for name,value in zip(totals,values):totals[name]+=value
    if seen!=set(jobs):raise ValueError('missing output conversion counters')
    return totals


def verify_timing(proof, caps, stages, rows, cols, core_ps):
    if proof is None:
        return False
    if not (proof.get('timing_status') in ('array_closed','kernel_closed') and proof.get('scope') == 'array'
            and caps.get('target','array') == 'array'
            and proof.get('rows') == rows and proof.get('cols') == cols
            and proof.get('stages') == stages and proof.get('period_ns', float('inf')) <= core_ps / 1000
            and proof.get('rtl_sha256') == caps['rtl_sha256']):
        raise ValueError('timing evidence does not cover this exact simulated implementation')
    if not(proof.get('setup_wns_ns',-1)>=0 and proof.get('hold_wns_ns',-1)>=0 and proof.get('tns_ns',-1)==0):
        raise ValueError('timing slacks do not close at the declared period')
    for key in ('electrical_clean', 'coverage_clean', 'route_drc_clean', 'setup_hold_closed'):
        if proof.get(key) is not True:
            raise ValueError(f'incomplete signoff: {key}')
    evidence_names=['netlist', 'sdc', 'spef', 'setup', 'hold', 'electrical', 'coverage', 'coverage_analysis', 'drc']
    if proof.get('constraint_policy_version',0)>=2:evidence_names.append('constraint_analysis')
    for name in evidence_names:
        path = Path(proof.get('evidence', {}).get(name, ''))
        digest = proof.get('evidence_sha256', {}).get(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'missing or changed timing evidence: {name}')
    if proof['evidence_sha256']['netlist']!=proof.get('netlist_sha256')or proof['evidence_sha256']['sdc']!=proof.get('sdc_sha256'):
        raise ValueError('timing identity differs from authenticated evidence')
    if proof.get('constraint_policy_version',0)>=2:
        from .constraints import core_policy
        if not core_policy(Path(proof['evidence']['sdc']).read_text(),proof['period_ns'])['clean']:
            raise ValueError('core clock/IO constraints do not match the supported complete timing policy')
    import re
    if re.search(r'set_false_path[^\n]*\breset\b',Path(proof['evidence']['sdc']).read_text()):
        raise ValueError('synchronous core reset paths were excluded from timing')
    if proof.get('constraint_policy_clean') is False:
        raise ValueError('physical constraint policy is incomplete')
    return True


def compare(baseline, candidate):
    required = ['workload_sha256', 'stages', 'rows', 'cols', 'rtl_sha256',
                'simulator_sha256','testbench_sha256','dram_library_sha256','dma_backend',
                'core_frequency_hz', 'memory_frequency_hz', 'dram', 'm', 'n', 'k']
    if baseline.get('dram'):required+=['dram_config_sha256']
    if 'input_schedule'in baseline or'input_schedule'in candidate:required+=['input_schedule']
    for key in required:
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            raise ValueError(f'incomparable experiments: {key}')
    if baseline['workload_sha256'] is None:
        raise ValueError('comparison requires a shared, identified input workload')
    for report in (baseline, candidate):
        if not (report.get('exact_output_match') or report.get('exact_hardware_reference_match')):
            raise ValueError('numerical verification missing')
        if report['elapsed_ps'] <= 0:
            raise ValueError('invalid elapsed time')
    validated = baseline['timing_validated'] and candidate['timing_validated']
    if 'cell_area_um2'in baseline or'cell_area_um2'in candidate:
        for key in('cell_area_um2','area_scope','area_netlist_sha256','area_liberty_sha256'):
            if baseline.get(key)!=candidate.get(key)or key not in baseline:
                raise ValueError(f'incomparable area evidence: {key}')
    result = {
        'baseline_format': baseline['format'], 'candidate_format': candidate['format'],
        'baseline_operand_mode':baseline.get('operand_mode','symmetric'),
        'candidate_operand_mode':candidate.get('operand_mode','symmetric'),
        'speedup': baseline['elapsed_ps'] / candidate['elapsed_ps'],
        'baseline_elapsed_ps': baseline['elapsed_ps'],
        'candidate_elapsed_ps': candidate['elapsed_ps'],
        'time_kind': 'timing_validated' if validated else 'configured_clock_projection',
        'timing_validated': bool(validated),
        'workload_sha256': baseline['workload_sha256'],
        'rtl_sha256': baseline['rtl_sha256'],
        'core_frequency_hz': baseline['core_frequency_hz'],
    }
    result.update({key:baseline[key]for key in required if key not in result})
    fields=('trace_sha256','source_npz_sha256','compute_cycles','dequant_operations',
            'dequant_memory_cycles','auxiliary_scalar_operations','group_auxiliary_lanes',
            'logical_input_bytes','logical_scale_bytes','dram_physical_bytes','dma_transfer_bytes',
            'onchip_moved_bytes','useful_bandwidth_GB_s','dram_bandwidth_GB_s','gmac_s',
            'relative_l2','framework_relative_l2','cell_area_um2','area_netlist_sha256',
            'area_scope','area_kind','area_report_sha256','area_statistics_sha256','area_liberty_sha256','area_mapping_profile','gops_per_mm2',
            'sequencer_cycle_breakdown','sequencer_memory_cycles','sequencer_cycle_clock',
            'area_full_storage_capacity','area_retained_memory_bytes','dma_bus_bandwidth_GB_s',
            'dma_peak_per_direction_GB_s','dma_read_bus_utilization','dma_write_bus_utilization',
            'output_format','output_scale_bits','output_scale_exact','output_zero_point',
            'output_arithmetic','output_error_domain','epilogue_elements',
            'epilogue_auxiliary_vectors','epilogue_memory_cycles','epilogue_saturations',
            'output_payload_bytes','logical_output_bytes')
    result['baseline_metrics']={key:baseline[key]for key in fields if key in baseline}
    result['candidate_metrics']={key:candidate[key]for key in fields if key in candidate}
    return result
