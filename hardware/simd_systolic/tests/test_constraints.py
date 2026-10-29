import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval.constraints import core_policy


class CoreConstraintPolicyTest(unittest.TestCase):
    def test_all_paths_and_actual_clock_must_remain_constrained(self):
        text='''create_clock -name core -period 1.000000 [get_ports clock]
set_clock_uncertainty 0.050 [get_clocks core]
set data_inputs [lsearch -all -inline -not -exact [all_inputs] [get_ports clock]]
set_input_delay 0.100 -clock core $data_inputs
set_input_transition 0.050 $data_inputs
set_output_delay 0.100 -clock core [all_outputs]
set_load 3.898 [all_outputs]
'''
        self.assertTrue(core_policy(text,1)['clean'])
        self.assertFalse(core_policy(text,2)['clean'])
        for extra in ['set_false_path -from [get_ports reset]',
                      'set_multicycle_path 2 -setup -through [get_pins *]',
                      'set_case_analysis 0 [get_ports io_fmt*]',
                      'set_disable_timing [get_cells *]',
                      'set_timing_derate -late 0.5',
                      'set_input_delay 0 -clock core $data_inputs']:
            self.assertFalse(core_policy(text+extra+'\n',1)['clean'],extra)
        self.assertFalse(core_policy(text.replace('0.050 [get_clocks','0.010 [get_clocks'),1)['clean'])
        self.assertFalse(core_policy(text.replace(' -exact [all_inputs]',' -exact [get_ports io_data]'),1)['clean'])


if __name__=='__main__':unittest.main()
