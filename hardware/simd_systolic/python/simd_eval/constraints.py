"""Audit the supported single-clock reference-core SDC without executing Tcl.

This intentionally recognizes the complete generated policy. Other policies,
including multiclock systems, need a separate audit and cannot inherit a core
certificate merely because STA has no unconstrained-endpoint warning.
"""
import re
from decimal import Decimal


def core_policy(sdc,period):
    lines=[re.sub(r'\s+',' ',line.strip()) for line in sdc.splitlines()
           if line.strip() and not line.lstrip().startswith('#')]
    explicit_units=bool(lines and lines[0]=='set_cmd_units -time ns -capacitance fF')
    if explicit_units:lines=lines[1:]
    expected=[
        r'create_clock -name core -period ([0-9]+(?:\.[0-9]+)?) \[get_ports clock\]',
        r'set_clock_uncertainty 0\.050 \[get_clocks core\]',
        r'set data_inputs \[lsearch -all -inline -not -exact \[all_inputs\] \[get_ports clock\]\]',
        r'set_input_delay 0\.100 -clock core \$data_inputs',
        r'set_input_transition 0\.050 \$data_inputs',
        r'set_output_delay 0\.100 -clock core \[all_outputs\]',
        r'set_load 3\.898 \[all_outputs\]',
    ]
    problems=[];actual_period=None
    if len(lines)!=len(expected):problems.append('SDC must contain exactly the complete seven-command core policy')
    for index,pattern in enumerate(expected):
        match=re.fullmatch(pattern,lines[index])if index<len(lines)else None
        if match is None:
            problems.append(f'unsupported or missing policy command {index+1}')
        elif index==0:
            actual_period=Decimal(match[1])
            if actual_period!=Decimal(str(period)):
                problems.append('SDC clock period differs from the reported analysis period')
    return dict(clean=not problems,policy_version=3,scope='single_clock_core',explicit_units=explicit_units,
                period_ns=float(actual_period)if actual_period is not None else None,
                required_setup_hold_uncertainty_ns=.05,required_input_delay_ns=.1,required_output_delay_ns=.1,
                required_input_transition_ns=.05,required_output_load_fF=3.898,
                all_data_inputs_including_synchronous_reset=not problems,
                all_outputs_constrained=not problems,timing_exceptions_permitted=False,
                problems=problems)


def system_policy(sdc,expected=None):
    """Only the regenerated, netlist-specific dual-clock policy is accepted."""
    problems=[]
    if expected is None or sdc!=expected:problems.append('SDC differs from netlist-specific dual-clock policy')
    for forbidden in ('set_clock_groups','set_multicycle_path','set_disable_timing','set_case_analysis'):
        if forbidden in sdc:problems.append('forbidden timing override: '+forbidden)
    if 'set_cmd_units -time ns -capacitance fF' not in sdc:problems.append('explicit units missing')
    for clock,period,port in [('core','1.000000','io_coreClock'),('memory','2.000000','io_memoryClock')]:
        if f'create_clock -name {clock} -period {period} [get_ports {port}]' not in sdc:
            problems.append('missing target clock: '+clock)
    if sdc.count('\nsimd_constrain_gray ')!=16:problems.append('incomplete FIFO Gray bus constraints')
    return dict(clean=not problems,scope='dual_clock_system',policy_version=1,
                core_period_ns=1,memory_period_ns=2,problems=problems)
