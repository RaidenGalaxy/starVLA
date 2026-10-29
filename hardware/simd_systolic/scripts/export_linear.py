#!/usr/bin/env python3
"""Export real Linear inputs/weights and provenance without downloading a model.

Array mode consumes activation [*, K] and PyTorch weight [N, K] .npy files.
Qwen-cache mode reconstructs attention or FFN inputs from saved FP32 decoder
boundaries. FFN reconstruction is checked against its captured module output.
Hardware input rounding happens later, in the selected precision evaluator.
Run Qwen-cache mode with the repository's torch/safetensors Python environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def export(args):
    if any(value is not None and value<1 for value in(args.m,args.n,args.k)):
        raise ValueError('crop dimensions must be positive')
    evidence = {'schema_version': 1, 'phase': args.phase, 'layer': args.layer}
    if args.cache:
        import torch
        import torch.nn.functional as functional
        from safetensors import safe_open
        torch.set_num_threads(args.threads)
        if args.layer_index < 1:
            raise ValueError('cache reconstruction requires layer_index >= 1')
        if args.phase=='decode':
            raise ValueError('teacher-forced cache is not an incremental decode trace; import a captured decode activation array')
        config = json.loads((args.model / 'config.json').read_text())
        text_config = config.get('text_config', config)
        epsilon = text_config.get('rms_norm_eps', 1e-6)
        index = json.loads((args.model / 'model.safetensors.index.json').read_text())['weight_map']
        # Transformers checkpoints use either of these prefixes across versions.
        projection=args.projection
        path='self_attn.q_proj'if projection=='q_proj'else'mlp.'+projection
        suffix = f'layers.{args.layer_index}.{path}.weight'
        names = [name for name in index if name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f'cannot uniquely locate {suffix}')
        weight_name = names[0]
        norm_name = weight_name.replace(path+'.weight',
            'input_layernorm.weight'if projection=='q_proj'else'post_attention_layernorm.weight')
        def tensor(name):
            with safe_open(str(args.model / index[name]), framework='pt', device='cpu') as f:
                return f.get_tensor(name)
        cache = torch.load(args.cache, map_location='cpu', weights_only=True)
        dtype=getattr(torch,args.cache_dtype)
        hidden = cache['hidden'][args.layer_index - 1].to(dtype)
        if projection!='q_proj':
            hidden=hidden+cache['boundaries']['decoder_attention'][args.layer_index].to(dtype)
        norm = tensor(norm_name).to(dtype)
        x = hidden.float()
        x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)).to(hidden.dtype)
        x = norm * x
        reconstruction={}
        if projection!='q_proj':
            if text_config.get('hidden_act','silu')!='silu':raise ValueError('FFN reconstruction currently supports SiLU')
            prefix=weight_name.rsplit('mlp.',1)[0]+'mlp.'
            weights={name:tensor(prefix+name+'.weight').to(dtype)for name in('gate_proj','up_proj','down_proj')}
            gated=functional.silu(functional.linear(x,weights['gate_proj']))*functional.linear(x,weights['up_proj'])
            reconstructed=functional.linear(gated,weights['down_proj'])
            captured=cache['boundaries']['decoder_ffn'][args.layer_index].to(dtype)
            error=(reconstructed.float()-captured.float()).double()
            relative=float(error.norm()/captured.double().norm().clamp_min(1e-300))
            limit=1e-5 if dtype==torch.float32 else 1e-2
            reconstruction=dict(ffn_reconstruction_relative_l2=relative,ffn_reconstruction_limit=limit,
                ffn_reconstruction_passed=relative<=limit,
                ffn_projection_weight_sha256={name:hashlib.sha256(value.float().numpy().tobytes()).hexdigest()
                    for name,value in weights.items()})
            if relative>limit:raise ValueError(f'cache/model FFN reconstruction mismatch: {relative} > {limit}')
            if projection=='down_proj':x=gated
        a = x.reshape(-1, x.shape[-1]).float().numpy()
        weight = tensor(weight_name).float().numpy()
        evidence.update(layer=weight_name.removesuffix('.weight'), sample_id=cache.get('id'),
                        source='cached_boundaries_then_projection_reconstruction', cache_sha256=sha(args.cache),
                        model_config_sha256=sha(args.model / 'config.json'),
                        weight_tensor_sha256=hashlib.sha256(weight.tobytes()).hexdigest(),
                        activation_dtype=str(dtype),rms_norm_epsilon=epsilon,projection=projection,
                        cache_compute_dtype=args.cache_dtype,cache_phase='teacher_forced_full_forward',**reconstruction)
    else:
        if not args.activations or not args.weight:
            raise ValueError('provide --activations and --weight, or --cache and --model')
        a = np.load(args.activations, allow_pickle=False)
        weight = np.load(args.weight, allow_pickle=False)
        a = a.reshape(-1, a.shape[-1])
        evidence.update(source='captured_linear_tensors', activation_sha256=sha(args.activations),
                        weight_sha256=sha(args.weight))
    if a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[1]:
        raise ValueError('expected activations [M,K] and PyTorch weight [N,K]')
    if min(a.shape[0],a.shape[1],weight.shape[0])<1:raise ValueError('empty projection')
    full = [a.shape[0], weight.shape[0], a.shape[1]]
    m, n, k = [min(total, limit) if limit else total for total, limit in zip(full, [args.m, args.n, args.k])]
    a = np.asarray(a[:m, :k], dtype=np.float32)
    b = np.asarray(weight[:n, :k].T, dtype=np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('quantization export requires finite tensors')
    evidence.update(full_shape_mnk=full, evaluated_shape_mnk=[m, n, k], cropped=[m, n, k] != full,
                    bias_included=False, reduction_order='increasing K')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Keep a source-framework projection separate from the strict increasing-K
    # FMA oracle. A cropped K recomputes that exported projection explicitly.
    if args.cache:
        with torch.inference_mode():
            reference=functional.linear(torch.from_numpy(a.copy()),torch.from_numpy(b.T.copy())).numpy()
        evidence['reference_output_model']='PyTorch FP32 linear on exported projection, no bias'
    else:
        reference=a@b
        evidence['reference_output_model']='NumPy FP32 matmul on exported projection, no bias'
    np.savez(args.out, a=a, b=b, reference_output=reference)
    args.out.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('activations', 'weight', 'cache', 'model', 'out'):
        p.add_argument('--' + name, type=Path, required=name == 'out')
    p.add_argument('--layer', default='unknown')
    p.add_argument('--phase', choices=['prefill', 'decode', 'layer'], default='layer')
    p.add_argument('--layer-index', type=int, default=1)
    p.add_argument('--projection',choices=['q_proj','gate_proj','up_proj','down_proj'],default='q_proj')
    p.add_argument('--cache-dtype',choices=['float32','bfloat16'],default='float32')
    p.add_argument('--threads',type=int,default=1)
    for name in ('m', 'n', 'k'):
        p.add_argument('--' + name, type=int)
    export(p.parse_args())
