#!/usr/bin/env python3
"""Exercise external raw codes/scales through autonomous RTL descriptors."""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
from check_grouped import run,ROOT


def check(build_dir,out):
    from runtime import validated_build
    guard,caps=validated_build(build_dir)
    out.mkdir(parents=True,exist_ok=True);reports=[];rng=random.Random(1937)
    # Original tensors deliberately disagree with imported encodings. Silently
    # requantizing them would produce an all-zero result and fail this test.
    for name,width in [('I4',4),('I8',8),('F4',4),('F8',8)]:
        m,n,k,group=3,7,17,4;groups=(k+group-1)//group
        path=out/(name+'.npz')
        qa=np.array([[rng.getrandbits(width)for _ in range(k)]for _ in range(m)],dtype=np.uint8)
        qb=np.array([[rng.getrandbits(width)for _ in range(n)]for _ in range(k)],dtype=np.uint8)
        # Include nonuniform scales, zero scale, and subnormal scale bits.
        scales=[0,1,0x3f000000,0x3f800000,0x40000000]
        sa=np.array([[scales[(i+g)%len(scales)]for g in range(groups)]for i in range(m)],dtype=np.uint32)
        sb=np.array([[scales[(j+2*g+1)%len(scales)]for j in range(n)]for g in range(groups)],dtype=np.uint32)
        np.savez(path,a=np.zeros((m,k)),b=np.zeros((k,n)),qa=qa,qb=qb,
                 sa_bits=sa,sb_bits=sb,format=np.asarray(name),group_size=np.asarray(group,dtype=np.uint32))
        metrics=run(stages=caps['stages'],rows=caps['rows'],cols=caps['cols'],
            build_dir=build_dir,dram=True,out=out/name,quantized_npz=path)
        assert metrics['passed']and metrics['quantized_codes_imported']
        assert metrics['quantized_checkpoint_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        assert metrics['group_size']==group and metrics['macs']==m*n*k
        writes=[line.split()[2]for line in (out/name/'simulation.log').read_text().splitlines()if line.startswith('W ')]
        assert any(int(value)for value in writes),'encoded data was replaced by all-zero original tensors'
        reports.append(dict(format=name,path=str((out/name/'metrics.json').resolve())))
    result=dict(passed=True,scope='external_encoded_checkpoint_native_descriptors',
        original_tensors_deliberately_differ=True,formats=[r['format']for r in reports],
        nonuniform_zero_subnormal_scales=True,tail_group_and_matrix=True,reports=reports,
        rtl_sha256=caps['rtl_sha256'],simulator_sha256=caps['simulator_sha256'])
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    check(args.build_dir.resolve(),args.out.resolve())
