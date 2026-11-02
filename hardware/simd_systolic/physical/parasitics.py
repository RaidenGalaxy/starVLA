"""Check actual SPEF read diagnostics and complete routed-driver annotation."""
import re
import hashlib
import json
from pathlib import Path


def unannotated_names(log):
    names=set()
    for section in re.findall(r'SIMD_SPEF_READ_BEGIN\n(.*?)SIMD_SPEF_READ_END',log,re.S):
        match=re.search(r'Found (\d+) unannotated drivers\.\n(.*?)Found \d+ partially unannotated drivers\.',section,re.S)
        if not match:raise ValueError('missing complete driver list')
        found=[line.strip()for line in match[2].splitlines()if line.strip()]
        if len(found)!=int(match[1])or len(set(found))!=len(found):raise ValueError('unannotated driver list/count differs')
        names.update(found)
    return names


def audit(log,spef,driver_audit=None):
    problems=[];sections=re.findall(r'SIMD_SPEF_READ_BEGIN\n(.*?)SIMD_SPEF_READ_END',log,re.S)
    allowed=set()
    if driver_audit is not None:
        proof=json.loads(Path(driver_audit).read_text())
        if not proof['passed']or not proof['no_timing_exceptions']:raise ValueError('invalid zero-fanout audit')
        for filename,digest in proof['evidence'].items():
            with Path(filename).open('rb')as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=digest:raise ValueError('zero-fanout audit evidence changed: '+filename)
        with Path(spef).open('rb')as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=proof['source_spef_sha256']:raise ValueError('zero-fanout audit is for another SPEF')
        for line in Path(proof['driver_table']).read_text().splitlines()[1:]:
            name,classification,sinks,routed=line.split('\t')
            if classification not in('unconnected_output','zero_fanout_no_routing')or sinks!='0'or routed!='0':
                raise ValueError('coverage waiver includes a connected or routed driver')
            allowed.add(bytes.fromhex(name).decode())
    if not sections or log.count('SIMD_SPEF_READ_BEGIN')!=len(sections)or log.count('SIMD_SPEF_READ_END')!=len(sections):
        problems.append('missing or incomplete explicit SPEF read/annotation section')
    counts=[]
    for section in sections:
        if re.search(r'\[(?:WARNING|ERROR)\b',section):problems.append('tool warning/error during SPEF read or annotation audit')
        plain=re.findall(r'Found (\d+) unannotated drivers\.',section)
        partial=re.findall(r'Found (\d+) partially unannotated drivers\.',section)
        if len(plain)!=1 or len(partial)!=1:problems.append('missing unannotated-driver coverage counts')
        else:
            counts.append(dict(unannotated_drivers=int(plain[0]),partially_unannotated_drivers=int(partial[0])))
            try:missing=unannotated_names('SIMD_SPEF_READ_BEGIN\n'+section+'SIMD_SPEF_READ_END')
            except ValueError:missing=set();problems.append('missing complete unannotated driver list')
            if int(partial[0])or missing-allowed or(int(plain[0])and not missing):
                problems.append('routed driver parasitics are incompletely annotated')
    if re.search(r'\[WARNING STA-1657\]',log):problems.append('SPEF parser reported a syntax error')
    starts=set();ends=0;active=False;name_map=False
    with Path(spef).open()as stream:
        for number,line in enumerate(stream,1):
            text=line.strip()
            if text=='*NAME_MAP':name_map=True;continue
            if name_map:
                if text.startswith(('*PORTS','*D_NET')):name_map=False
                elif text:
                    if not re.fullmatch(r'\*\d+\s+\S+',text)or(len(text)-len(text.rstrip('\\')))%2:
                        problems.append('malformed SPEF name map at line '+str(number))
            if text.startswith('*D_NET '):
                fields=text.split()
                if len(fields)!=3:problems.append('malformed D_NET record at line '+str(number));continue
                if active or fields[1]in starts:problems.append('duplicate or unterminated SPEF net')
                try:float(fields[2])
                except ValueError:problems.append('invalid SPEF net capacitance')
                starts.add(fields[1]);active=True
            elif text=='*END':
                if not active:problems.append('SPEF END outside a net')
                ends+=1;active=False
    if not starts or active or ends!=len(starts):problems.append('SPEF lacks complete nonempty routed-net records')
    return dict(clean=not problems,read_sections=len(sections),annotation_counts=counts,
        structurally_proven_zero_fanout_drivers=len(allowed),
        spef_nets=len(starts),spef_end_records=ends,problems=sorted(set(problems)))
