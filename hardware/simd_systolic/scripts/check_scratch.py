#!/usr/bin/env python3
"""Exercise actual storage capacity, byte masks, errors, and stalled responses."""
import argparse,json,random,subprocess
from pathlib import Path
from runtime import validated_build
ROOT=Path(__file__).resolve().parents[1]
import sys as _architecture_sys
_architecture_sys.path.insert(0,str(ROOT/'python'))
from simd_eval.architecture import CORE_PS, MEMORY_PS

def check(args):
    directory=args.build_dir.resolve();guard,caps=validated_build(directory)
    if caps['target']!='gemm'or not caps.get('scratchpad_access'):raise ValueError('a GEMM build with full scratch access is required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(30871);commands=[];expected={};contents={};reads=writes=aggregate_reads=aggregate_writes=0
    tag=0;depths=[1024,2048,1024,8192]
    def issue(bank,address,write=False,value=0,strobe=0,error=False):
        nonlocal tag,reads,writes,aggregate_reads,aggregate_writes
        commands.append(f'C {bank} {address} {int(write)} {strobe:x} {value:032x} {tag}\n')
        key=bank,address;bits=64 if bank==3 else 128;mask=(1<<bits)-1
        if error:actual=0
        elif write:
            old=contents.get(key,0)
            for byte in range(bits//8):
                if strobe>>byte&1:old=(old&~(255<<(8*byte)))|(value&(255<<(8*byte)))
            contents[key]=old&mask;actual=0
            if bank==3:
                if strobe:aggregate_writes+=8
                if strobe and strobe!=255:aggregate_reads+=8
            else:
                writes+=sum(8 for word in range(2)if (strobe>>(8*word))&255)
                for word in range(2):
                    submask=(strobe>>(8*word))&255
                    if submask not in (0,255):reads+=8
        else:
            actual=contents[key]
            if bank==3:aggregate_reads+=8
            else:reads+=16
        expected[tag]=(int(error),actual);tag+=1
    for bank,depth in enumerate(depths):
        addresses=list(range(depth))if args.full else sorted({0,1,depth-2,depth-1,*[rng.randrange(depth)for _ in range(12)]})
        width=8 if bank==3 else 16
        for address in addresses:issue(bank,address,True,rng.getrandbits(width*8),(1<<width)-1)
        for address in reversed(addresses):issue(bank,address)
        for address in addresses[:min(32,len(addresses))]:
            for strobe in [0,1,(1<<width)-1,0x55 if bank==3 else 0x5555]:
                issue(bank,address,True,rng.getrandbits(width*8),strobe);issue(bank,address)
        if bank!=3:issue(bank,depth,error=True);issue(bank,depth,True,(1<<128)-1,65535,error=True)
    issue(3,8191,True,1<<100,256,error=True);issue(3,8191)
    # Revisit every bank after all other banks have changed, catching
    # cross-bank aliases as well as aliases within one address range.
    for bank,address in reversed(list(contents)):issue(bank,address)
    trace=out/'commands.txt';trace.write_text(''.join(commands))
    proc=subprocess.run([str(directory/'sim'),str(trace),str(CORE_PS),str(MEMORY_PS),'-',str(out/'dram'),'207'],
        capture_output=True,text=True,check=True,timeout=args.timeout)
    (out/'simulation.log').write_text(proc.stdout+proc.stderr);seen=set();onchip=None
    for line in proc.stdout.splitlines():
        if line.startswith('C '):
            _,number,error,value=line.split();number=int(number)
            if number in seen or (int(error),int(value,16))!=expected[number]:raise AssertionError('scratch data/error/tag mismatch')
            seen.add(number)
        if line.startswith('H '):onchip=list(map(int,line.split()[1:]))
    if seen!=set(expected)or onchip!=[reads,writes,aggregate_reads,aggregate_writes]:raise AssertionError('scratch coverage or byte accounting mismatch')
    report=dict(passed=True,kind='scratch_capacity_and_masks',full_address_sweep=args.full,transactions=len(seen),
        capacity_bytes=dict(A=16384,B=32768,C=16384,aggregate=65536),all_banks=True,
        invalid_requests=7,byte_strobes=True,response_backpressure_memory_clocks=7,
        noninteger_clock_ratio=False,sram_read_latency=caps['sram_read_latency'],rtl_sha256=caps['rtl_sha256'],
        simulator_sha256=caps['simulator_sha256'],onchip_bytes=onchip)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'artifacts/scratch')
    p.add_argument('--full',action='store_true');p.add_argument('--timeout',type=int,default=3600);check(p.parse_args())
