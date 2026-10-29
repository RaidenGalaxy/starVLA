"""Pinned ASAP7 views and a lossless merged standard-cell Liberty library."""
import fnmatch
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDK = ROOT / '.deps/asap7'
MACRO = 'srambank_256x4x64_6t122'


def block_end(text, opening):
    depth=1; i=opening+1; quoted=False
    while i<len(text):
        if not quoted and text.startswith('/*',i):
            i=text.index('*/',i+2)+2;continue
        char=text[i]
        if char=='\\' and quoted:i+=2;continue
        if char=='"':quoted=not quoted
        elif not quoted:
            if char=='{':depth+=1
            elif char=='}':
                depth-=1
                if not depth:return i+1
        i+=1
    raise ValueError('unterminated Liberty group')


def groups(text, pattern):
    for match in re.finditer(pattern+r'\s*\(\s*([^)]*?)\s*\)\s*\{',text):
        yield match[1].strip('"'),match.start(),block_end(text,match.end()-1)


def merged_library(corner='TT'):
    if corner not in ('TT','SS','FF'):raise ValueError('unsupported ASAP7 corner')
    lock=json.loads((ROOT/'dependencies.lock.json').read_text())['asap7']['files']
    inputs=sorted(PDK/name for name in lock if name.startswith('lib/NLDM/') and '_RVT_'+corner+'_' in name)
    if len(inputs)!=5:raise ValueError('ASAP7 views missing; run scripts/prepare_asap7.py')
    expected={str(p.relative_to(PDK)):hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs}
    if any(lock[name]['sha256']!=digest for name,digest in expected.items()):raise ValueError('ASAP7 source library changed')
    path=PDK/f'asap7_RVT_{corner}.lib';manifest=PDK/f'asap7_RVT_{corner}.manifest.json'
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if path.exists() and manifest.exists():
        cached=json.loads(manifest.read_text())
        if cached.get('generator_sha256')==generator_sha256 and cached.get('inputs')==expected and cached.get('sha256')==hashlib.sha256(path.read_bytes()).hexdigest():return path
    source=[];evidence={}
    for source_path in inputs:
        relative=str(source_path.relative_to(PDK));digest=hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest!=lock[relative]['sha256']:raise ValueError('ASAP7 library changed: '+relative)
        raw=gzip.decompress(source_path.read_bytes()).decode() if source_path.suffix=='.gz' else source_path.read_text()
        tag=source_path.name.split('_')[1]
        # Distinct template namespaces retain each source's exact defaults.
        for name,_,_ in list(groups(raw,r'(?:lu_table_template|power_lut_template)')):
            raw=re.sub(r'\b'+re.escape(name)+r'\b',tag+'_'+name,raw)
        source.append(raw);evidence[relative]=digest
    first=source[0];cell_groups=list(groups(first,r'\bcell'))
    header=first[:cell_groups[0][1]]
    extra=[];cells=[]
    for index,raw in enumerate(source):
        if index:
            extra.extend(raw[start:end]for _,start,end in groups(raw,r'(?:lu_table_template|power_lut_template)'))
        cells.extend(raw[start:end]for _,start,end in groups(raw,r'\bcell'))
    header=re.sub(r'library\s*\([^)]+\)',f'library (asap7_RVT_{corner})',header,count=1)
    result=header+'\n'+'\n'.join(extra+cells)+'\n}\n'
    if not path.exists() or path.read_text()!=result:path.write_text(result)
    (PDK/f'asap7_RVT_{corner}.manifest.json').write_text(json.dumps(dict(
        inputs=evidence,sha256=hashlib.sha256(result.encode()).hexdigest(),generator_sha256=generator_sha256,
        time_unit='ps',capacitance_unit='fF',physical_length_unit='um'),indent=2)+'\n')
    return path


def excluded_cells(liberty):
    patterns=('*x1p*_ASAP7*','*xp*_ASAP7*','SDF*','ICG*','TAP*','FILL*','DECAP*')
    return [name for name,_,_ in groups(liberty,r'\bcell')
            if any(fnmatch.fnmatchcase(name,p)for p in patterns)]


def profile(corner='TT'):
    return dict(root=PDK,liberty=merged_library(corner),
        tech_lef=PDK/'lef/asap7_tech_1x_201209.lef',
        cell_lef=PDK/'lef/asap7sc7p5t_28_R_1x_220121a.lef',
        tracks=PDK/'openRoad/make_tracks.tcl',rcx=PDK/'rcx_patterns.rules',
        site='asap7sc7p5t',tiehi='TIEHIx1_ASAP7_75t_R/H',tielo='TIELOx1_ASAP7_75t_R/L',
        cts_root='BUFx8_ASAP7_75t_R',cts_buffers=['BUFx2_ASAP7_75t_R','BUFx4_ASAP7_75t_R','BUFx8_ASAP7_75t_R'])
