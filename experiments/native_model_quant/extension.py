"""Build and load the optional AVX-512 VNNI PyTorch extension."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

from . import KERNEL_FORMAT_VERSION

PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE = PACKAGE_DIR / "csrc" / "native_quant.cpp"
DEFAULT_BUILD_DIR = Path("outputs/native_model_quant/torch_extensions").resolve()


def source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(KERNEL_FORMAT_VERSION.encode())
    digest.update(SOURCE.read_bytes())
    return digest.hexdigest()


def build_identity() -> dict[str, Any]:
    return {
        "format_version": KERNEL_FORMAT_VERSION,
        "source_hash": source_hash(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "compiler": os.environ.get("CXX", "c++"),
        "extra_cflags": ["-O3", "-std=c++17", "-DNDEBUG"],
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }


@lru_cache(maxsize=2)
def load_native_extension(build_directory: str | Path = DEFAULT_BUILD_DIR, *, verbose: bool = False):
    build_directory = Path(build_directory).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    environment_bin = str(Path(sys.executable).absolute().parent)
    if environment_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = environment_bin + os.pathsep + os.environ.get("PATH", "")
    name = f"starvla_native_quant_{source_hash()[:12]}"
    return load(
        name=name,
        sources=[str(SOURCE)],
        extra_cflags=["-O3", "-std=c++17", "-DNDEBUG"],
        build_directory=str(build_directory),
        verbose=verbose,
        with_cuda=False,
    )


def require_avx512_vnni(extension=None) -> None:
    extension = extension or load_native_extension()
    if not bool(extension.supports_avx512_vnni()):
        raise RuntimeError("avx512_w4a8 requires AVX-512 VNNI; no floating fallback is permitted")


def write_build_manifest(path: Path, extension=None) -> dict[str, Any]:
    extension = extension or load_native_extension()
    payload = {
        **build_identity(),
        "avx512_vnni": bool(extension.supports_avx512_vnni()),
        "extension_name": extension.__name__,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload
