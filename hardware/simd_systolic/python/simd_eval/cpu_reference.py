"""Optional independently compiled host FMA reference; never supplies RTL data."""
import ctypes,fcntl,hashlib,json,subprocess
from pathlib import Path
import numpy as np

_loaded=None
def load():
    global _loaded
    if _loaded is not None:return _loaded
    source=Path(__file__).with_suffix('.cc');root=source.parents[2]
    compiler=subprocess.check_output(['g++','--version'],text=True)
    contents=source.read_bytes();target=subprocess.check_output(['g++','-dumpmachine'],text=True).strip()
    flags=['-shared','-fPIC','-O2','-std=c++17','-frounding-math','-fno-fast-math','-ffp-contract=off']
    identity=hashlib.sha256(contents+compiler.encode()+target.encode()+json.dumps(flags).encode()).hexdigest()
    directory=root/'.build/cpu_reference'/identity[:16];directory.mkdir(parents=True,exist_ok=True)
    library=directory/'reference.so'
    with(directory/'.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        frozen=directory/'reference.cc';frozen.write_bytes(contents)
        if not library.exists():subprocess.run(['g++',*flags,str(frozen),'-o',str(library),'-lm'],check=True)
    evidence=dict(reference_backend='host_cpu_fma_rne_sequential_k',reference_compiler=compiler.splitlines()[0],
        reference_source_sha256=hashlib.sha256(contents).hexdigest(),reference_compile_target=target,reference_compile_flags=flags,
        reference_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    (directory/'manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    api=ctypes.CDLL(str(library));function=api.simd_fma_gemm
    u64=np.ctypeslib.ndpointer(dtype=np.uint64,ndim=2,flags='C_CONTIGUOUS')
    u32=np.ctypeslib.ndpointer(dtype=np.uint32,ndim=2,flags='C_CONTIGUOUS')
    function.argtypes=[u64,u64,ctypes.c_void_p,u32,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,u32]
    function.restype=ctypes.c_int;_loaded=(function,evidence,api);return _loaded

def gemm(a,b,fmt,initial=None):
    aa=np.ascontiguousarray(a,dtype=np.uint64);bb=np.ascontiguousarray(b,dtype=np.uint64)
    if aa.ndim!=2 or bb.ndim!=2 or aa.shape[1]!=bb.shape[0]or min(*aa.shape,*bb.shape)<1:
        raise ValueError('expected nonempty encoded A[M,K], B[K,N]')
    m,k=aa.shape;n=bb.shape[1];output=np.empty((m,n),dtype=np.uint32)
    cc=None if initial is None else np.ascontiguousarray(initial,dtype=np.uint32)
    if cc is not None and cc.shape!=(m,n):raise ValueError('initial accumulator shape differs')
    function,evidence,api=load();status=np.empty((m,n),dtype=np.uint32)
    error=function(aa,bb,None if cc is None else ctypes.c_void_p(cc.ctypes.data),output,m,n,k,int(fmt),status)
    if error:raise ValueError('CPU reference supports FP4/FP8/FP16/BF16/FP32 inputs with FP32 accumulation')
    return output.tolist(),status.tolist(),dict(evidence)
