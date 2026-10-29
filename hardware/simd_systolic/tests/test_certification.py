import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from simd_eval.certification import core_certificate,compatible_core_support
from simd_eval.architecture import ARCH


class CoreCertificateTest(unittest.TestCase):
    def test_routed_numbers_complete_constraints_and_every_evidence_hash_required(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            files={name:root/(name+'.txt')for name in ['netlist','sdc','spef','setup','hold','electrical','coverage','coverage_analysis','drc']}
            for name,path in files.items():path.write_text(name)
            files['sdc'].write_text('''set_cmd_units -time ns -capacitance fF
create_clock -name core -period 1.000000 [get_ports clock]
set_clock_uncertainty 0.050 [get_clocks core]
set data_inputs [lsearch -all -inline -not -exact [all_inputs] [get_ports clock]]
set_input_delay 0.100 -clock core $data_inputs
set_input_transition 0.050 $data_inputs
set_output_delay 0.100 -clock core [all_outputs]
set_load 3.898 [all_outputs]
''')
            flow=root/'flow.tcl';flow.write_text('reference flow')
            library=root/'cells.lib';library.write_text('reference library')
            proof=dict(process='asap7',architecture_sha256=ARCH['architecture_sha256'],scope='array',stages=2,rows=2,cols=2,period_ns=1,
                setup_wns_ns=.01,hold_wns_ns=.02,tns_ns=0,timing_status='kernel_closed',
                setup_hold_closed=True,electrical_clean=True,coverage_clean=True,
                route_drc_clean=True,constraint_policy_clean=True,rtl_sha256='rtl',
                evidence={k:str(p)for k,p in files.items()},evidence_sha256={k:sha(p)for k,p in files.items()},
                netlist_sha256=sha(files['netlist']),sdc_sha256=sha(files['sdc']),
                flow_sha256=sha(flow),liberty_file=str(library),liberty_sha256=sha(library))
            numeric=dict(passed=True,scope='array',stages=2,rows=2,cols=2,
                mapped_netlist_kind='routed',all_formats=True,rtl_sha256='rtl',
                wave_drain_protocol_checked=True,all_wave_stages_tracked=True,
                netlist_sha256=proof['netlist_sha256'],liberty_sha256=proof['liberty_sha256'])
            self.assertTrue(core_certificate(proof,numeric)['passed'])
            for field,value in [('all_formats',False),('mapped_netlist_kind','synthesis'),
                                ('netlist_sha256','different'),('rows',8),
                                ('wave_drain_protocol_checked',False),('all_wave_stages_tracked',False)]:
                altered=dict(numeric,**{field:value})
                self.assertFalse(core_certificate(proof,altered)['passed'],field)
            for field,value in [('hold_wns_ns',-.001),('setup_wns_ns',float('nan')),
                                ('electrical_clean',False),('period_ns',.9)]:
                self.assertFalse(core_certificate(dict(proof,**{field:value}),numeric)['passed'],field)
            missing=copy.deepcopy(proof);del missing['evidence']['spef']
            self.assertFalse(core_certificate(missing,numeric)['passed'])
            files['spef'].write_text('modified extraction')
            self.assertFalse(core_certificate(proof,numeric)['passed'])

    def test_pe_and_subarray_must_share_f2_arithmetic_support(self):
        manifest={name:name for name in ['ControlReplicas.sv','PrefixAdd.sv','fpnew_pkg.sv',
                   'fpnew_fma.sv','SimdFpF2.sv','SimdFpF2Core.sv']}
        pe=dict(stages=2,architecture_sha256=ARCH['architecture_sha256'],input_manifest=manifest)
        self.assertTrue(compatible_core_support(pe,copy.deepcopy(pe)))
        other=copy.deepcopy(pe);other['input_manifest']['ControlReplicas.sv']='different'
        self.assertFalse(compatible_core_support(pe,other))
        self.assertFalse(compatible_core_support(pe,dict(pe,stages=8)))


if __name__=='__main__':unittest.main()
