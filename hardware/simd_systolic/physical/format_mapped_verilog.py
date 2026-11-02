"""Wrap huge Yosys netlist lines using existing whitespace only."""
from pathlib import Path


def wrap_assignments(path,width=4096,all_statements=False):
    path=Path(path);lines=[];changed=0
    for line in path.read_text().splitlines(keepends=True):
        if len(line)<=width or(not all_statements and not line.lstrip().startswith('assign ')):
            lines.append(line);continue
        if '"'in line or '//'in line or '/*'in line:
            raise ValueError('refuse to reformat a statement containing quotes/comments')
        tokens=line.split();chunks=[];current='  '
        for token in tokens:
            if len(current)+len(token)+1>width and current.strip():
                chunks.append(current.rstrip());current='    '
            current+=token+' '
        chunks.append(current.rstrip());wrapped='\n'.join(chunks)+'\n'
        # Whitespace still terminates every escaped identifier, and every
        # non-whitespace token is byte-identical to the original assignment.
        if wrapped.split()!=tokens:raise ValueError('Verilog token sequence changed')
        lines.append(wrapped);changed+=1
    path.write_text(''.join(lines));return changed
