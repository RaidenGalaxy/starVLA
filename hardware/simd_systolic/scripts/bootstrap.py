#!/usr/bin/env python3
"""Fetch checksummed local dependencies; do not modify system packages."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def fetch(url, dest, digest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temporary = dest.with_suffix(dest.suffix + '.part')
        print('Downloading', url, flush=True)
        # GitHub's release endpoints can be rate limited; curl retries and
        # resumes the local partial download without touching unrelated files.
        subprocess.run(['curl', '-fL', '--retry', '3', '-o', str(temporary), url], check=True)
        temporary.replace(dest)
    actual = hashlib.sha256(dest.read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f'checksum mismatch: {dest}: {actual}')


def bootstrap(physical=False):
    lock = json.loads((ROOT / 'dependencies.lock.json').read_text())
    for name in ('java', 'sbt', 'verilator', 'cmake', 'g++', 'git'):
        if not shutil.which(name):
            raise RuntimeError(f'{name} required; see README.md or Dockerfile')
    if physical:
        tools = ROOT / '.deps/tools'
        for name, item in lock['downloads'].items():
            fetch(item['url'], tools / name, item['sha256'])
        if not (tools / 'oss-cad-suite/bin/yosys').exists():
            with tarfile.open(tools / 'oss.tgz') as archive:
                archive.extractall(tools, filter='data')
        for archive in ('openroad.deb', 'libpython310.deb'):
            installed=tools/'openroad'/('usr/bin/openroad'if archive=='openroad.deb'else'usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0')
            if installed.exists():continue
            subprocess.run(['dpkg-deb', '-x', str(tools / archive), str(tools / 'openroad')], check=True)
        from prepare_asap7 import prepare
        prepare()
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_dramsim.py')], check=True)
    print('Dependencies ready. Versions are recorded in dependencies.lock.json.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--physical', action='store_true')
    bootstrap(p.parse_args().physical)
