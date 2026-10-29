"""Hold a shared build lock for the entire simulation and authenticate its inputs."""
import fcntl
import hashlib
import json
from build import rtl_digest
from simd_eval.architecture import ARCH, require_f2


def validated_build(directory):
    guard = (directory / '.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_SH)
    if (directory/'.incomplete').exists():
        raise RuntimeError(f'incomplete or failed build in {directory}; rerun scripts/build.py')
    caps = json.loads((directory / 'capabilities.json').read_text())
    require_f2(caps['stages'])
    if caps.get('architecture_sha256')!=ARCH['architecture_sha256']:
        raise RuntimeError('historical architecture snapshot; rebuild for ASAP7 F2')
    root=directory
    from pathlib import Path
    project=Path(__file__).resolve().parents[1]
    for field,name in [('dram_config_sha256','configs/ddr4.ini'),('dependency_lock_sha256','dependencies.lock.json')]:
        if hashlib.sha256((project/name).read_bytes()).hexdigest()!=caps.get(field):
            raise RuntimeError('configuration/dependency identity changed; rebuild: '+name)
    manifest=json.loads((directory/'rtl_manifest.json').read_text())
    paths=[directory/'inputs'/name for name in manifest]
    if any(not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[path.name]for path in paths):
        raise RuntimeError(f'RTL source snapshot changed in {directory}')
    if rtl_digest(paths,directory/'inputs/common_cells')!=caps['rtl_sha256']:
        raise RuntimeError(f'RTL/header identity mismatch in {directory}')
    for name,digest in caps.get('testbench_manifest',{}).items():
        path=directory/'testbench'/name
        if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'testbench snapshot changed in {directory}: {name}')
    actual = hashlib.sha256((directory / 'sim').read_bytes()).hexdigest()
    if actual != caps.get('simulator_sha256'):
        raise RuntimeError(f'unverified/incomplete build in {directory}; rerun scripts/build.py')
    if 'dram_library_sha256' in caps:
        lib=directory/'libraries/libdramsim3.so'
        if not lib.is_file() or hashlib.sha256(lib.read_bytes()).hexdigest()!=caps['dram_library_sha256']:
            raise RuntimeError(f'DRAMSim3 library changed in {directory}; rerun scripts/build.py')
    return guard, caps
