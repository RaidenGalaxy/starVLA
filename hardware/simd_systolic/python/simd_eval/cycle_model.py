"""Standalone mesh timing from the public latency contract and trace protocol.

This model does not read RTL counters or measured cycles. It covers the
valid-only wave mesh and observation port, not autonomous DMA/DRAM scheduling.
"""


def mesh_trace_cycles(caps,commands):
    if caps['target']!='array':raise ValueError('standalone array capabilities required')
    rows,cols=caps['rows'],caps['cols'];hop=caps['hop_latency']
    read_latency=caps['array_read_latency'];release=caps.get('array_busy_release_latency',0)
    result_latency=caps['stages']+caps.get('pe_commit_latency',0)+caps.get('pe_state_read_latency',0)
    settle=caps.get('counter_settle_cycles',2)
    cycle=0;busy_until=0;pending=[];reads={};markers={};drains=[]

    def observe(tag,when):
        if tag in reads:raise ValueError('duplicate read tag')
        reads[tag]=when

    def advance(target):
        nonlocal cycle
        while pending and pending[0][1]<=target:
            tag,due=pending.pop(0);observe(tag,due)
        cycle=target

    for line in commands:
        fields=line.split()
        if not fields:continue
        command=fields[0]
        if command!='P'and pending:advance(pending[-1][1])
        if command=='I':
            row_mask=int(fields[5],16)
            if row_mask>>rows:raise ValueError('row mask outside array')
            # Every B token crosses all rows, including zero-mask columns.
            wave_drain=(rows+cols-1)*hop+1
            active=[row for row in range(rows)if row_mask&(1<<row)]
            output_drain=((max(active)+cols-1)*hop+result_latency+settle+1)if active else 0
            busy_until=max(busy_until,cycle+max(wave_drain,output_drain)+release)
            advance(cycle+1)
        elif command=='W':advance(cycle+int(fields[1]))
        elif command=='D':
            advance(max(cycle+1,busy_until));drains.append(cycle)
        elif command=='S':
            if cycle<busy_until:raise ValueError('seed while array busy')
            busy_until=cycle+caps.get('seed_write_latency',1)+release
            advance(cycle+3)
        elif command in('R','P'):
            tag=int(fields[4])
            if command=='R':advance(cycle+read_latency);observe(tag,cycle)
            elif read_latency:
                pending.append((tag,cycle+read_latency));advance(cycle+1)
            else:observe(tag,cycle);advance(cycle+1)
        elif command in('T','E'):
            key=(command,int(fields[1]))
            if key in markers:raise ValueError('duplicate marker')
            markers[key]=cycle
        else:raise ValueError('unsupported array trace command '+command)
    if pending:advance(pending[-1][1])
    return dict(reads=reads,markers=markers,drains=drains,final_cycles=cycle)


def verify_mesh_cycles(stdout,prediction):
    reads={};markers={}
    for line in stdout.splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='R':
            tag=int(fields[1])
            if tag in reads:raise AssertionError('duplicate observed read tag')
            reads[tag]=int(fields[4])
        elif fields[0]in('T','E'):
            key=(fields[0],int(fields[1]))
            if key in markers:raise AssertionError('duplicate observed marker')
            markers[key]=int(fields[2])
    if reads!=prediction['reads']:
        differences=[(tag,prediction['reads'].get(tag),reads.get(tag))
                     for tag in prediction['reads'].keys()|reads.keys()
                     if reads.get(tag)!=prediction['reads'].get(tag)]
        raise AssertionError('mesh read cycle mismatch: '+str(differences[:8]))
    if markers!=prediction['markers']:raise AssertionError('mesh marker cycle mismatch')
    return dict(reads=len(reads),markers=len(markers),drains=len(prediction['drains']),
                last_observation_cycle=max([*reads.values(),*markers.values()],default=0),
                final_bridge_cycle_prediction=prediction['final_cycles'])
