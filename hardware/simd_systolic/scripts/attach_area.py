#!/usr/bin/env python3
"""Add verified same-netlist area to an already completed layer evaluation."""
import argparse,json
from pathlib import Path
from benchmark_layer import annotate_area,write_comparisons

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--area',type=Path,required=True);args=parser.parse_args()
    out=args.directory.resolve();status=json.loads((out/'job_status.json').read_text())
    if not status or any(status.values())or'BF16'not in status:raise ValueError('layer jobs have not all completed successfully')
    modes=[name for name in status if name!='BF16']
    annotate_area(out,args.area,modes);write_comparisons(out,modes)
    print(f'Authenticated complete-system area attached to {len(status)} precision reports in {out}')
