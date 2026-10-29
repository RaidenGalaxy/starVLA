#!/usr/bin/env python3
"""Capture an actual prefill and KV-cache decode Linear call on local CPU weights.

This optional workload tool needs the repository's torch/transformers environment.
It downloads nothing and writes only to its output directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):value.update(chunk)
    return value.hexdigest()


def capture(args):
    if not args.model.is_dir():raise ValueError('--model must be a local checkpoint directory')
    if args.layer_index<0 or min(args.m,args.n,args.threads)<1 or(args.k is not None and args.k<1):
        raise ValueError('invalid layer, shape or thread count')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    samples=[json.loads(line)for line in args.manifest.read_text().splitlines()if line.strip()]
    samples=[sample for sample in samples if sample['id']==args.sample_id]
    if len(samples)!=1:raise ValueError('sample id must identify exactly one manifest entry')
    sample=samples[0];image_path=Path(sample['image'])
    if not image_path.is_absolute():image_path=(args.manifest.parent/image_path).resolve()
    started=time.monotonic()
    print('Loading local checkpoint in FP32 on CPU.',flush=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(args.model),torch_dtype=torch.float32,
        attn_implementation='eager',low_cpu_mem_usage=True,local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(str(args.model),min_pixels=56*56,max_pixels=112*112,local_files_only=True)
    with torch.inference_mode():
        if args.reference=='fp16_rounded_fp32':
            for parameter in model.parameters():
                if not parameter.is_floating_point():continue
                half=parameter.detach().half()
                if not torch.isfinite(half).all():raise ValueError('FP16 reference weight conversion overflowed')
                parameter.data=half.float()
    suffix=f'layers.{args.layer_index}.mlp.{args.projection}'
    selected=[(name,module)for name,module in model.named_modules()if name.endswith(suffix)]
    if len(selected)!=1:raise ValueError('cannot uniquely locate decoder projection '+suffix)
    name,linear=selected[0]
    if not isinstance(linear,torch.nn.Linear)or linear.bias is not None:raise ValueError('expected a bias-free Linear projection')
    if any(p.is_floating_point()and p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('model is not FP32')
    with Image.open(image_path)as source:
        image=source.convert('RGB')
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':sample['prompt']}]}]
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[prompt],images=[image],padding=True,return_tensors='pt')
    inputs={key:value.to('cpu')for key,value in inputs.items()if torch.is_tensor(value)}
    context={};captured={};calls=[]
    def before_forward(_module,_args,kwargs):
        past=kwargs.get('past_key_values')
        if past is None:length=0
        elif hasattr(past,'get_seq_length'):length=int(past.get_seq_length())
        elif isinstance(past,(list,tuple)):
            length=int(past[0][0].shape[-2])if len(past)else 0
        else:raise ValueError('unknown KV cache type; decode provenance cannot be established')
        position=kwargs.get('cache_position')
        context.clear();context.update(past_tokens=length,
            cache_type=type(past).__name__,cache_position=position.detach().cpu().reshape(-1).tolist()if torch.is_tensor(position)else None)
    def linear_hook(_module,arguments,output):
        x=arguments[0];tokens=int(x.shape[-2]);phase='decode'if context['past_tokens']>0 else'prefill'
        if phase=='decode'and tokens!=1:raise ValueError('expected one-token incremental decode')
        calls.append(dict(phase=phase,linear_shape=list(x.shape),**context))
        if phase not in captured:
            captured[phase]=(x.detach().float().cpu().clone(),output.detach().float().cpu().clone(),dict(context))
    handles=[model.register_forward_pre_hook(before_forward,with_kwargs=True),linear.register_forward_hook(linear_hook)]
    try:
        with torch.inference_mode():
            generated=model.generate(**inputs,do_sample=False,min_new_tokens=2,max_new_tokens=2,use_cache=True,
                pad_token_id=getattr(processor.tokenizer,'pad_token_id',None))
    finally:
        for handle in handles:handle.remove()
    if set(captured)!={'prefill','decode'}:raise ValueError('generation did not produce both real prefill and cached decode calls')
    weight=linear.weight.detach().float().cpu().contiguous()
    weight_sha=hashlib.sha256(weight.numpy().tobytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=True);records=[]
    for phase,(activation,reference,provenance)in captured.items():
        activation=activation.reshape(-1,activation.shape[-1]);reference=reference.reshape(-1,reference.shape[-1])
        full=[activation.shape[0],weight.shape[0],activation.shape[1]]
        m=min(args.m,full[0]);n=min(args.n,full[1]);k=min(args.k,full[2])if args.k else full[2]
        a=activation[:m,:k].contiguous();b=weight[:n,:k].t().contiguous()
        if k==full[2]:expected=reference[:m,:n].contiguous();kind='captured PyTorch FP32 Linear output'
        else:
            with torch.inference_mode():expected=functional.linear(a,b.t())
            kind='PyTorch FP32 Linear recomputed for explicitly cropped K'
        if not all(torch.isfinite(tensor).all()for tensor in(a,b,expected)):raise ValueError('nonfinite captured data')
        path=args.out/f'{phase}.npz';np.savez(path,a=a.numpy(),b=b.numpy(),reference_output=expected.numpy())
        record=dict(schema_version=1,phase=phase,layer=name,source='actual_model_generate_linear_hook',
            sample_id=args.sample_id,reference=args.reference,compute_dtype='torch.float32',device='cpu',
            full_shape_mnk=full,evaluated_shape_mnk=[m,n,k],cropped=full!=[m,n,k],
            bias_included=False,reduction_order='increasing K',reference_output_model=kind,
            kv_cache=provenance,weight_tensor_sha256=weight_sha,manifest_sha256=sha(args.manifest),
            image_sha256=sha(image_path),model_config_sha256=sha(args.model/'config.json'),
            npz_sha256=sha(path),prompt_tokens=int(inputs['input_ids'].shape[1]),
            prompt_input_ids_sha256=hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
    report=dict(passed=True,captured_phases=['prefill','decode'],generation_calls=calls,
        generated_token_ids=generated[0,inputs['input_ids'].shape[1]:].tolist(),
        model_path=str(args.model.resolve()),host_elapsed_seconds=time.monotonic()-started,
        software_versions={name:importlib.metadata.version(name)for name in('torch','transformers','numpy','Pillow')},
        workloads=records)
    (args.out/'capture.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--sample-id',default='eval_0000')
    parser.add_argument('--projection',choices=['gate_proj','up_proj','down_proj'],default='down_proj')
    parser.add_argument('--layer-index',type=int,default=1);parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--m',type=int,default=3);parser.add_argument('--n',type=int,default=19);parser.add_argument('--k',type=int)
    parser.add_argument('--reference',choices=['fp16_rounded_fp32','checkpoint_fp32'],default='fp16_rounded_fp32')
    parser.add_argument('--out',type=Path,required=True);capture(parser.parse_args())
