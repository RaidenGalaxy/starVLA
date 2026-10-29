#!/usr/bin/env python3
"""Fill and reuse every grouped output-context slot through real descriptors."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from check_grouped import run
from runtime import validated_build


def check(build,out,timeout=14400):
    if timeout<=0:raise ValueError('host timeout must be positive')
    guard,caps=validated_build(build)
    rows,cols=caps['rows'],caps['cols'];n=cols*16*caps['contexts'];k=2
    out.mkdir(parents=True,exist_ok=True);path=out/'codes.npz'
    # Raw INT4 values alone repeat after sixteen columns. Distinct FP32
    # scales make every first-group lane unique across rows, PEs and contexts,
    # so truncating any aggregate address bit cannot pass by reading an equal
    # value from a different slot. Powers of two make this first product exact.
    sa=np.array([[2.0**(12*r),2.0**r]for r in range(rows)],dtype=np.float32)
    sb=np.stack([np.arange(1,n+1),np.arange(n,0,-1)]).astype(np.float32)
    qa=np.ones((rows,2),dtype=np.uint8);qb=np.ones((2,n),dtype=np.uint8)
    qb[1,1::2]=15
    a=sa.copy();b=sb.copy();b[1,1::2]*=-1
    first=a[:,0,None]*b[None,0,:]
    assert np.unique(first.view(np.uint32)).size==rows*n
    np.savez(path,a=a,b=b,qa=qa,qb=qb,
        sa_bits=sa.view(np.uint32),sb_bits=sb.view(np.uint32),
        format=np.asarray('I4'),group_size=np.asarray(1,dtype=np.uint32))
    try:
        metrics=run(stages=caps['stages'],rows=rows,cols=cols,dram=True,build_dir=build,
            quantized_npz=path,out=out/'run',timeout=timeout)
    except subprocess.TimeoutExpired:
        (out/'verification_failed.json').write_text(json.dumps(dict(passed=False,status='host_timeout',
            numerical_failure_observed=False,timeout_seconds=timeout,rows=rows,cols=cols,
            rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256']),indent=2)+'\n')
        raise
    slots=rows*n//2
    assert metrics['aggregate_read_bytes']==8*slots
    assert metrics['aggregate_write_bytes']==16*slots
    assert metrics['macs']==rows*n*k and metrics['exact_output_match']
    report=dict(passed=True,scope='native_grouped_context_capacity_boundary',rows=rows,cols=cols,
        all_contexts=caps['contexts'],groups=2,highest_aggregate_index=slots-1,
        distinct_context_slot_reuse_checked=True,all_outputs_verified=True,
        unique_first_group_lanes=rows*n,nonuniform_group_scales=True,
        aggregate_read_bytes=metrics['aggregate_read_bytes'],aggregate_write_bytes=metrics['aggregate_write_bytes'],
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--timeout',type=int,default=14400)
    args=parser.parse_args();check(args.build_dir.resolve(),args.out.resolve(),args.timeout)
