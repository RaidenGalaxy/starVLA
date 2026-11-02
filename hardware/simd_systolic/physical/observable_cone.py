"""Exact structural certificate for deleting cells outside all output cones.

Sequential cells are traversed through every input, including clock/reset.
No reachable cell, connection, parameter or initialized state may change.
"""
from collections import defaultdict


def network(module):
    drivers=defaultdict(list);inputs=defaultdict(list)
    for name,port in module['ports'].items():
        if port['direction']in('input','inout'):
            for index,bit in enumerate(port['bits']):inputs[bit].append(('input',name,index))
    for name,cell in module['cells'].items():
        for port,bits in cell['connections'].items():
            if port not in cell['port_directions']:raise ValueError('cell port has no direction')
            if cell['port_directions'][port]in('output','inout'):
                for index,bit in enumerate(bits):drivers[bit].append(('cell',name,port,index))
    return drivers,inputs


def live_cells(module):
    drivers,_=network(module)
    todo=[bit for port in module['ports'].values()if port['direction']in('output','inout')for bit in port['bits']]
    bits=set();cells=set()
    while todo:
        bit=todo.pop()
        if not isinstance(bit,int)or bit in bits:continue
        bits.add(bit)
        for _,name,_,_ in drivers.get(bit,[]):
            if name in cells:continue
            cells.add(name);cell=module['cells'][name]
            for port,values in cell['connections'].items():
                if cell['port_directions'][port]in('input','inout'):todo.extend(values)
    return cells,bits


def signatures(module):
    drivers,inputs=network(module)
    def bit_signature(bit):
        if isinstance(bit,str):return ('constant',bit)
        values=inputs.get(bit,[])+drivers.get(bit,[])
        if not values:raise ValueError('reachable signal has no driver or input anchor')
        return tuple(sorted(values))
    return bit_signature


def certify(before,after,top):
    records=[];todo=[top];visited=set()
    while todo:
        name=todo.pop()
        if name in visited:continue
        visited.add(name)
        if name not in before['modules']or name not in after['modules']:
            raise ValueError('reachable module removed: '+name)
        old,new=before['modules'][name],after['modules'][name]
        if old.get('attributes',{}).get('blackbox','').endswith('1'):
            if old!=new:raise ValueError('library blackbox changed: '+name)
            continue
        old_sig,new_sig=signatures(old),signatures(new)
        live,live_bits=live_cells(old)
        if set(new['cells'])!=live:raise ValueError('cleanup did not preserve exactly the observable cell cone: '+name)
        if set(old['ports'])!=set(new['ports']):raise ValueError('module interface changed')
        for port,data in old['ports'].items():
            other=new['ports'][port]
            if data['direction']!=other['direction']or [old_sig(b)for b in data['bits']]!=[new_sig(b)for b in other['bits']]:
                raise ValueError('observable port binding changed: '+port)
        for cell_name in live:
            a,b=old['cells'][cell_name],new['cells'][cell_name]
            if any(a.get(k)!=b.get(k)for k in('type','parameters','port_directions')):
                raise ValueError('observable cell type/parameters changed: '+cell_name)
            if set(a['connections'])!=set(b['connections']):raise ValueError('cell ports changed')
            for port,bits in a['connections'].items():
                if [old_sig(x)for x in bits]!=[new_sig(x)for x in b['connections'][port]]:
                    raise ValueError('observable connection changed: '+cell_name+'/'+port)
            if a['type']in before['modules']:todo.append(a['type'])
        def initialized(module,signature,only=None):
            result=defaultdict(set)
            for net in module['netnames'].values():
                value=net.get('attributes',{}).get('init')
                if value is None:continue
                value=value.zfill(len(net['bits']))[::-1]
                for bit,initial in zip(net['bits'],value):
                    if only is None or bit in only:result[signature(bit)].add(initial)
            return dict(result)
        _,new_live_bits=live_cells(new)
        if initialized(old,old_sig,live_bits)!=initialized(new,new_sig,new_live_bits):
            raise ValueError('observable initialized state changed')
        records.append(dict(module=name,before_cells=len(old['cells']),after_cells=len(new['cells']),
            removed_unobservable_cells=len(old['cells'])-len(new['cells'])))
    return dict(passed=True,method='exact output-cone structural bisimulation including sequential inputs',
        modules=records,all_observable_ports_and_cell_connections_identical=True,retiming_performed=False)
