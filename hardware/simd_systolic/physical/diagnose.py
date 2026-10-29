#!/usr/bin/env python3
"""Resolve numeric STA cell names to retained RTL state for timing iteration."""
import argparse,hashlib,json,re
from pathlib import Path

def diagnose(directory,routed=False):
    net=directory/('routed.v'if routed else'synth.v');report=directory/('setup.rpt'if routed else'preopt_setup.rpt')
    if not report.exists():report=directory/'setup.rpt';net=directory/'routed.v'
    paths=[]
    for section in re.split(r'(?=Startpoint:)',report.read_text()):
        start=re.search(r'Startpoint: (\S+)',section);end=re.search(r'Endpoint: (\S+)',section)
        slack=re.search(r'([-\d.]+)\s+slack',section)
        if start and end and slack:paths.append(dict(startpoint=start[1],endpoint=end[1],slack_ns=float(slack[1])))
    names={part for p in paths for key in('startpoint','endpoint')for part in p[key].split('/')}
    names.update('/'.join(p[key].split('/')[i:])for p in paths for key in('startpoint','endpoint')for i in range(len(p[key].split('/'))))
    modules={}
    if names:
        pattern=r'(\S+)\s+\\?('+ '|'.join(map(re.escape,names))+r')\s*\((.*?)\);'
        for module in re.finditer(r'^module\s+(\S+)\s*\(.*?^endmodule',net.read_text(),re.S|re.M):
            cells={}
            for match in re.finditer(pattern,module[0],re.S):
                pins={k:v.strip().lstrip('\\')for k,v in re.findall(r'\.(\w+)\s*\(([^)]+)\)',match[3])}
                cells[match[2]]=dict(cell_type=match[1].lstrip('\\'),q=pins.get('Q',pins.get('QN')),d=pins.get('D'))
            modules[module[1].lstrip('\\')]=cells
    top=json.loads((directory/'build_identity.json').read_text())['top']
    def resolve(path):
        owner=top;parts=path.split('/')
        for index,instance in enumerate(parts[:-1]):
            flattened=modules.get(owner,{}).get('/'.join(parts[index:]))
            if flattened is not None:return flattened
            cell=modules.get(owner,{}).get(instance)
            if cell is None:return None
            owner=cell['cell_type']
        return modules.get(owner,{}).get(parts[-1])
    for path in paths:
        for key in('startpoint','endpoint'):
            path[key+'_state']=resolve(path[key])
    result=dict(report=str(report),netlist_sha256=hashlib.sha256(net.read_bytes()).hexdigest(),
                evidence_kind='cts_path_diagnosis'if report.name.startswith('preopt')else'post_route_path_diagnosis',paths=paths)
    (directory/('routed_path_diagnosis.json'if routed else'path_diagnosis.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--routed',action='store_true')
    args=parser.parse_args();diagnose(args.directory.resolve(),args.routed)
