#!/usr/bin/env python3
"""Isolated F2 PE context-read experiment, with an exact combinational proof.

The active frozen workspace stays unchanged. A source copy gets sixteen local
32-bit context selectors; shared context wires no longer drive the full binary
8192-bit mux. Optional FPU inputs come from an authenticated component snapshot.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

SCALA='''class F2ContextWordRead extends BlackBox {
  override def desiredName="SimdF2ContextWordRead"
  val io=IO(new Bundle {val ctx=Input(UInt(4.W));val words=Input(UInt(512.W));val value=Output(UInt(32.W))})
}
'''
RTL='''
// Combinational local read: no new register or arithmetic stage.
(* keep_hierarchy = "yes" *)
module SimdF2ContextWordRead(input wire [3:0] ctx,input wire [511:0] words,output wire [31:0] value);
 wire [15:0] selected=16'b1<<ctx;
 wire [31:0] tree[1:31];
 for(genvar i=0;i<16;i++) assign tree[16+i]=words[32*i+:32]&{32{selected[i]}};
 for(genvar i=1;i<16;i++) assign tree[i]=tree[2*i]|tree[2*i+1];
 assign value=tree[1];
endmodule
'''

INTEGER_CONVERSION='''object IntegerToFloat {
  def apply(x:UInt,w:Int,stages:Int,ce:Bool):(UInt,UInt) = {
    require(stages==2)
    val sign=x(w-1)
    val mag=Mux(sign,(-x.asSInt).asUInt,x)
    val s=RegEnable(sign,ce);val m=RegEnable(mag,ce)
    val choices=(0 until w).map {j=>
      val selected=m(j) && (if(j==w-1)true.B else !m(w-1,j+1).orR)
      val keep=if(j>=23)m(j,j-23) else Cat(m(j,0),0.U((23-j).W))
      val guard=if(j>=24)m(j-24) else false.B
      val sticky=if(j>=25)m(j-25,0).orR else false.B
      selected->Cat((j+127).U(9.W),keep,guard,sticky)
    }
    val selected=Mux1H(choices)
    val exponent=selected(34,26);val keep=selected(25,2)
    val guard=selected(1);val sticky=selected(0)
    val rounded=keep+&(guard && (sticky || keep(0)))
    val exp=exponent+rounded(24);val overflow=exp>=255.U
    val zero= !m.orR
    val data=Mux(zero,0.U(32.W),Mux(overflow,Cat(s,255.U(8.W),0.U(23.W)),
      Cat(s,exp(7,0),Mux(rounded(24),rounded(23,1),rounded(22,0)))))
    val flags=Mux(zero,0.U(5.W),Cat(0.U(2.W),overflow,false.B,guard||sticky||overflow))
    (RegEnable(data,ce),RegEnable(flags,ce))
  }
}
'''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.registered_integer_prefixes and not args.integer_conversion_onehot:
        raise ValueError('registered integer prefixes require the onehot conversion experiment')
    if args.balanced_integer_select and not args.integer_conversion_onehot:
        raise ValueError('balanced integer selection requires the onehot conversion experiment')
    out=args.out.resolve();work=args.work.resolve()
    if out.exists()or work.exists():raise ValueError('preserve prior experiments; choose new --out and --work')
    out.mkdir(parents=True);work.mkdir(parents=True)
    shutil.copy2(__file__,out/'experiment_driver.py')
    for folder in ('src','rtl','vendor','python','scripts','physical','configs','sim','tests','project'):
        shutil.copytree(ROOT/folder,work/folder,ignore=shutil.ignore_patterns('__pycache__','target','*.pyc'))
    for filename in ('Makefile','build.sbt','dependencies.lock.json'):
        shutil.copy2(ROOT/filename,work/filename)
    (work/'.deps').symlink_to(ROOT/'.deps',target_is_directory=True)
    array=work/'src/main/scala/simd/Array.scala';before=sha(array);raw=array.read_text()
    marker='class SimdPE(cfg:Config,streaming:Boolean=false)'
    if raw.count(marker)!=1:raise ValueError('unexpected PE declaration')
    raw=raw.replace(marker,SCALA+'\n'+marker)
    old='  if(cfg.stages==2)raw.acc:=Mux(io.in.bits.clear || !initialized(io.in.bits.ctx),0.U,accum(io.in.bits.ctx))'
    if raw.count(old)!=1:raise ValueError('unexpected F2 context read')
    new='''  if(cfg.stages==2) {
    val readWords=(0 until 16).map {word=>
      val local=Module(new F2ContextWordRead)
      local.io.ctx:=io.in.bits.ctx
      local.io.words:=Cat((0 until cfg.contexts).reverse.map(i=>accum(i)(32*word+31,32*word)))
      local.io.value
    }
    raw.acc:=Mux(io.in.bits.clear || !initialized(io.in.bits.ctx),0.U,Cat(readWords.reverse))
  }'''
    raw=raw.replace(old,new)
    if args.all_read_ports:
        old='    io.readData:=Mux(initialized(io.readCtx),accum(io.readCtx),0.U);io.readFlags:=flags(io.readCtx)'
        if raw.count(old)!=1:raise ValueError('unexpected F2 observation read')
        raw=raw.replace(old,'''    val observationWords=(0 until 16).map {word=>
      val local=Module(new F2ContextWordRead)
      local.io.ctx:=io.readCtx
      local.io.words:=Cat((0 until cfg.contexts).reverse.map(i=>accum(i)(32*word+31,32*word)))
      local.io.value
    }
    io.readData:=Mux(initialized(io.readCtx),Cat(observationWords.reverse),0.U)
    io.readFlags:=flags(io.readCtx)''')
    array.write_text(raw)
    if args.local_format:
        arithmetic=work/'src/main/scala/simd/Arithmetic.scala';text=arithmetic.read_text()
        begin=text.index('class SimdAlu(');head,body=text[:begin],text[begin:]
        old='  val valid=RegInit(VecInit(Seq.fill(cfg.stages)(false.B)))'
        if body.count(old)!=1:raise ValueError('unexpected ALU validity declaration')
        body=body.replace(old,'''  val sourceFormats=Module(new SourceFormatReplicas)
  sourceFormats.io.fmt:=io.in.bits.fmt
  def sourceFormat(lane:Int,f:Int):Bool=sourceFormats.io.value(11*lane+f)
'''+old)
        for old,new in [
            ('val active=io.in.bits.fmt===f.U && io.in.bits.mask(lane)',
             'val active=sourceFormat(lane,f) && io.in.bits.mask(lane)'),
            ('when(io.in.bits.fmt===f.U && io.in.bits.mask(lane))',
             'when(sourceFormat(lane,f) && io.in.bits.mask(lane))')]:
            if body.count(old)!=1:raise ValueError('unexpected ALU source format gate')
            body=body.replace(old,new)
        selected='io.in.bits.fmt>=5.U && io.in.bits.fmt<=9.U && Formats.mask(io.in.bits.fmt)(lane) && io.in.bits.mask(lane)'
        if body.count(selected)!=2:raise ValueError('unexpected FP32 lane selection')
        local='(5 to 9).filter(f=>lane<64/Formats.widths(f)).map(f=>sourceFormat(lane,f)).reduce(_||_) && io.in.bits.mask(lane)'
        body=body.replace('val selected='+selected,'val selected='+local)
        body=body.replace('Mux('+selected+',','Mux('+local+',')
        if body.count('io.in.bits.fmt===10.U')!=1:raise ValueError('unexpected FP64 source selection')
        body=body.replace('io.in.bits.fmt===10.U','sourceFormat(15,10)')
        arithmetic.write_text(head+body)
    if args.integer_conversion_onehot:
        arithmetic=work/'src/main/scala/simd/Arithmetic.scala';text=arithmetic.read_text()
        start=text.index('object IntegerToFloat {');end=text.index('/** F2 uses an elastic pipeline;',start)
        conversion=INTEGER_CONVERSION
        if args.registered_integer_prefixes:
            declaration='''class F2IntegerPrefixes(w:Int) extends BlackBox {
  override def desiredName=s"SimdF2IntegerPrefixes$w"
  val io=IO(new Bundle {val clock=Input(Clock());val ce=Input(Bool());val value=Input(UInt(w.W))
    val leading=Output(UInt(w.W));val lower=Output(UInt(w.W))})
}
'''
            conversion=declaration+conversion.replace('stages:Int,ce:Bool):(UInt,UInt)',
                'stages:Int,ce:Bool,clk:Clock):(UInt,UInt)').replace('    val s=RegEnable(sign,ce);val m=RegEnable(mag,ce)',
                '''    val s=RegEnable(sign,ce);val m=RegEnable(mag,ce)
    val prefixes=Module(new F2IntegerPrefixes(w))
    prefixes.io.clock:=clk;prefixes.io.ce:=ce;prefixes.io.value:=mag''').replace(
                'val selected=m(j) && (if(j==w-1)true.B else !m(w-1,j+1).orR)',
                'val selected=prefixes.io.leading(j)').replace(
                'val sticky=if(j>=25)m(j-25,0).orR else false.B',
                'val sticky=if(j>=25)prefixes.io.lower(j-25) else false.B')
            conversion=conversion.replace('val zero= !m.orR','val zero= !prefixes.io.lower(w-1)')
        if args.balanced_integer_select:
            conversion='''class F2IntegerSelect(w:Int) extends BlackBox {
  override def desiredName=s"SimdF2IntegerSelect$w"
  val io=IO(new Bundle {val candidates=Input(UInt((35*w).W));val enable=Input(UInt(w.W))
    val result=Output(UInt(35.W))})
}
'''+conversion.replace('    val selected=Mux1H(choices)',
                '''    val selection=Module(new F2IntegerSelect(w))
    selection.io.candidates:=Cat(choices.reverse.map(_._2))
    selection.io.enable:=Cat(choices.reverse.map(_._1))
    val selected=selection.io.result''')
        tail=text[end:]
        if args.registered_integer_prefixes:
            if tail.count('aw,cfg.stages,ce)')!=1:raise ValueError('unexpected conversion call site')
            tail=tail.replace('aw,cfg.stages,ce)','aw,cfg.stages,ce,clock)')
        arithmetic.write_text(text[:start]+conversion+'\n'+tail)
    if args.integer_fast_sum:
        arithmetic=work/'src/main/scala/simd/Arithmetic.scala';text=arithmetic.read_text()
        for old,new in[
            ('prod.pad(aw+1)+old.pad(aw+1)',
             'FastCarry.add(prod.pad(aw+1).asUInt,old.pad(aw+1).asUInt,aw+1).asSInt '),
            ('aa.pad(aw+1)+bb.pad(aw+1)',
             'FastCarry.add(aa.pad(aw+1).asUInt,bb.pad(aw+1).asUInt,aw+1).asSInt '),
            ('aa.pad(aw+1)-bb.pad(aw+1)',
             'FastCarry.add(aa.pad(aw+1).asUInt,~bb.pad(aw+1).asUInt,aw+1,true.B).asSInt ')]:
            if text.count(old)!=1:raise ValueError('unexpected F2 integer sum')
            text=text.replace(old,new)
        arithmetic.write_text(text)
    controls=work/'rtl/ControlReplicas.sv';controls.write_text(controls.read_text()+RTL)
    if args.balanced_integer_select:
        for width in(32,48,80,144):
            with controls.open('a')as stream:stream.write(f'''
// Preserve a logarithmic result selection tree; no register is added.
(* keep_hierarchy=1 *)
module SimdF2IntegerSelect{width}(input wire [{35*width-1}:0] candidates,
 input wire [{width-1}:0] enable,output wire [34:0] result);
 localparam W={width},N=1<<$clog2(W);
 (* keep *) wire [34:0] tree[1:2*N-1];
 for(genvar i=0;i<N;i++)begin
   if(i<W)assign tree[N+i]=candidates[35*i+:35]&{{35{{enable[i]}}}};
   else assign tree[N+i]=35'b0;
 end
 for(genvar i=1;i<N;i++)assign tree[i]=tree[2*i]|tree[2*i+1];
 assign result=tree[1];
endmodule
''')
    if args.registered_integer_prefixes:
        for width in(32,48,80,144):
            with controls.open('a')as stream:stream.write(f'''
// Parallel with the existing magnitude FF: no third arithmetic stage.
(* keep_hierarchy=1 *)
module SimdF2IntegerPrefixes{width}(input wire clock,ce,input wire [{width-1}:0] value,
 output reg [{width-1}:0] leading,lower);
 localparam W={width},L=$clog2(W);
 (* keep *) wire [W-1:0] up[0:L],down[0:L];
 assign up[0]=value;assign down[0]=value;
 for(genvar level=1;level<=L;level++)begin
   for(genvar j=0;j<W;j++)begin
     if(j+(1<<(level-1))<W)assign up[level][j]=up[level-1][j]|up[level-1][j+(1<<(level-1))];
     else assign up[level][j]=up[level-1][j];
     if(j>=(1<<(level-1)))assign down[level][j]=down[level-1][j]|down[level-1][j-(1<<(level-1))];
     else assign down[level][j]=down[level-1][j];
   end
 end
 for(genvar j=0;j<W;j++)begin
   wire selected;
   if(j==W-1)assign selected=value[j];
   else assign selected=value[j]&&!up[L][j+1];
   always @(posedge clock)if(ce)begin leading[j]<=selected;lower[j]<=down[L][j];end
 end
endmodule
''')
    component=None
    if args.fpu_snapshot:
        parent=args.fpu_snapshot.resolve();identity=json.loads((parent/'build_identity.json').read_text())
        verified=json.loads((parent/'rtl_verification.json').read_text())
        if not verified['passed']or verified['rtl_sha256']!=identity['rtl_sha256']:
            raise ValueError('FPU snapshot has no matching numerical proof')
        component=dict(directory=str(parent),rtl_sha256=identity['rtl_sha256'],
            reference_scope=verified.get('scope'),sources={})
        for name in ('SimdFpF2.sv','SimdFpF2Core.sv','PrefixAdd.sv'):
            source=parent/'inputs'/name
            if sha(source)!=identity['input_manifest'][name]:raise ValueError('FPU input changed: '+name)
            shutil.copy2(source,work/'rtl'/name);component['sources'][name]=sha(source)
    report=dict(kind='isolated_f2_context_word_read_experiment',pipeline_stages=2,
        work_directory=str(work),main_source_changed=False,array_before_sha256=before,
        array_after_sha256=sha(array),control_rtl_sha256=sha(controls),
        fpu_component=component,steps=[],functional_passed=False,physical_frequency_certified=False)
    report['context_read_scope']='arithmetic and observation'if args.all_read_ports else'arithmetic'
    report['local_format_decoders']=args.local_format
    report['integer_conversion_onehot']=args.integer_conversion_onehot
    report['registered_integer_prefixes']=args.registered_integer_prefixes
    report['integer_fast_sum']=args.integer_fast_sum
    report['balanced_integer_select']=args.balanced_integer_select
    report['physical_target']=args.physical_target
    def save():(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    def call(name,command,cwd=work):
        log=out/(name+'.log');print(name,flush=True)
        with log.open('w')as stream:
            result=subprocess.run(list(map(str,command)),cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
        report['steps'].append(dict(name=name,command=list(map(str,command)),returncode=result.returncode,
            log=str(log),log_sha256=sha(log)));save()
        if result.returncode:raise RuntimeError('experiment failed at '+name+'; see '+str(log))
    save()
    miter=out/'context_miter.sv';miter.write_text('''module ContextReadMiter(input [3:0] ctx,input [511:0] words,output equivalent);
wire [31:0] actual;SimdF2ContextWordRead dut(.*,.value(actual));
wire [31:0] reference=words >> {ctx,5'b0};
assign equivalent=(actual==reference);
endmodule
''')
    proof=out/'context_proof.ys';proof.write_text(f'''read_verilog -sv {controls} {miter}
hierarchy -check -top ContextReadMiter
setattr -mod -unset keep_hierarchy SimdF2ContextWordRead
proc
flatten
opt_clean
sat -prove equivalent 1 -verify
''')
    call('context_equivalence',['yosys','-Q','-T','-s',proof])
    for kind in ('alu','pe'):
        call('build_'+kind,[sys.executable,'scripts/build.py','--kind',kind,'--stages','2','--rows','1',
            '--label','ctx_word','-j',str(args.jobs),*(['--hierarchical']if kind=='pe'else[])])
    alu=work/'.build/alu_2_1x1_ctx_word';pe=work/'.build/pe_2_1x1_ctx_word'
    call('alu_all_formats',[sys.executable,'scripts/check_alu.py','--build-dir',alu,'--exhaustive','--stall','--idle-gaps'])
    if args.integer_conversion_onehot:
        call('conversion_rounding_sweep',[sys.executable,ROOT/'physical/check_integer_conversion.py',
            '--build-dir',alu,'--out',out/'rounding_sweep'])
    for name in ('check_pe','check_pe_state','check_ii'):
        call(name,[sys.executable,'scripts/'+name+'.py','--build-dir',pe])
    report['functional_passed']=True;save()
    target=args.physical_target
    call('physical_cts',[sys.executable,'physical/run.py','--target',target,'--stages','2','--rows','1',
        '--period','1','--label','ctx_word','--rtl-build',pe if target=='pe'else alu,'--mapping-profile','fast','--cts-only'])
    timing=work/f'.build/physical/asap7_TT_{target}_2_1x1_1ns_ctx_word/cts_timing.json'
    report['timing']=dict(path=str(timing),sha256=sha(timing),data=json.loads(timing.read_text()));save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fpu-snapshot',type=Path);p.add_argument('-j','--jobs',type=int,default=8)
    p.add_argument('--all-read-ports',action='store_true',help='also localize the unchanged zero-latency observation read')
    p.add_argument('--local-format',action='store_true',help='reuse kept combinational local format decoders; no new register')
    p.add_argument('--integer-conversion-onehot',action='store_true',help='parallel fixed slices replace binary conversion shifters within F2')
    p.add_argument('--registered-integer-prefixes',action='store_true',help='capture balanced prefix trees at the existing first-stage boundary')
    p.add_argument('--integer-fast-sum',action='store_true',help='balanced exact integer add/sub/MAC carry trees inside the unchanged second stage')
    p.add_argument('--balanced-integer-select',action='store_true',help='preserve a logarithmic conversion result tree without adding registers')
    p.add_argument('--physical-target',choices=['pe','int64'],default='pe')
    run(p.parse_args())
