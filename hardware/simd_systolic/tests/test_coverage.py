import importlib.util
from pathlib import Path
import unittest
path=Path(__file__).resolve().parents[1]/'physical/coverage.py'
spec=importlib.util.spec_from_file_location('coverage_proof',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ConstraintCoverageTest(unittest.TestCase):
    def test_asap7_tie_pin_names_prove_only_constant_endpoints(self):
        net='''TIEHIx1_ASAP7_75t_R hi (.H(one));
        TIELOx1_ASAP7_75t_R lo (.L(zero));
        DFF_X1 first (.D(one),.Q(q1));
        DFF_X1 second (.D(zero),.Q(q2));
        DFF_X1 third (.D(data),.Q(q3));'''
        report='Warning: There are 2 unconstrained endpoints.\n first/D\n second/D\n'
        proof=module.analyze(net,report)
        self.assertTrue(proof['clean'])
        self.assertEqual([p['value']for p in proof['constant_endpoints']],[1,0])
        self.assertFalse(module.analyze(net,report.replace('second/D','third/D'))['clean'])

    library='''cell (NAND3_X1) {
      pin (A1) { direction : input; }
      pin (A2) { direction : input; }
      pin (A3) { direction : input; }
      pin (ZN) { direction : output; function : "!((A1 & A2) & A3)"; }
    }
    cell (DFF_X1) {
      ff (IQ, IQN) { next_state : "D"; clocked_on : "CK"; }
      pin (D) { direction : input; }
      pin (Q) { direction : output; function : "IQ"; }
    }'''

    def test_library_logic_matches_all_nand_input_combinations(self):
        report='Warning: There is 1 unconstrained endpoint.\n ff/D\n'
        for value in range(8):
            net='\n'.join(f'LOGIC{(value>>i)&1}_X1 tie{i} (.Z(n{i}));'for i in range(3))
            net+='\nNAND3_X1 gate (.A1(n0), .A2(n1), .A3(n2), .ZN(d));\nDFF_X1 ff (.D(d), .Q(q));'
            self.assertFalse(module.analyze(net,report)['clean'])
            result=module.analyze(net,report,self.library)
            self.assertTrue(result['clean'])
            self.assertEqual(result['constant_endpoints'][0]['value'],int(value!=7))
            self.assertEqual(len(result['constant_logic_proofs']['d']['inputs']),3)

    def test_library_logic_keeps_unknown_inputs_unproven(self):
        net='''LOGIC0_X1 tie (.Z(zero));
        NAND3_X1 gate (.A1(zero), .A2(zero), .A3(data), .ZN(d));
        DFF_X1 ff (.D(d), .Q(q));'''
        report='Warning: There is 1 unconstrained endpoint.\n ff/D\n'
        self.assertFalse(module.analyze(net,report,self.library)['clean'])

    def test_library_functions_never_propagate_through_sequential_cells(self):
        net='''LOGIC0_X1 tie (.Z(zero));
        DFF_X1 first (.D(zero), .Q(q));
        DFF_X1 second (.D(q), .Q(result));'''
        report='Warning: There is 1 unconstrained endpoint.\n second/D\n'
        self.assertFalse(module.analyze(net,report,self.library)['clean'])

    def test_only_tie_driven_endpoints_are_exempt(self):
        net='''LOGIC0_X1 tie (.Z(zero));
        BUF_X2 buf (.A(zero), .Z(buffered));
        INV_X1 inv (.A(buffered), .ZN(one));
        DFF_X1 ff0 (.D(buffered), .CK(clock), .Q(q0));
        DFF_X1 ff1 (.D(one), .CK(clock), .Q(q1));
        DFF_X1 ff2 (.D(data), .CK(clock), .Q(q2));'''
        report='Warning: There are 2 unconstrained endpoints.\n ff0/D\n ff1/D\n'
        result=module.analyze(net,report)
        self.assertTrue(result['clean'])
        self.assertEqual([p['value']for p in result['constant_endpoints']],[0,1])
        self.assertFalse(module.analyze(net,report.replace('ff1/D','ff2/D'))['clean'])
        self.assertFalse(module.analyze(net,report+'Warning: no clock\n')['clean'])
        self.assertFalse(module.analyze(net,report.replace('2 unconstrained','3 unconstrained'))['clean'])
        escaped=net.replace('ff1 (',r'\hier.ff1 (')
        self.assertTrue(module.analyze(escaped,report.replace('ff1/D','hier.ff1/D'))['clean'])

    def test_constant_output_is_proven_through_aliases(self):
        net='''module top(output [4:0] flags, output data);
        LOGIC0_X1 tie (.Z(zero));
        BUF_X1 buffer (.A(zero), .Z(buffered));
        assign alias = buffered;
        assign flags[3] = alias;
        assign flags[2] = data;
        endmodule'''
        report='Warning: There is 1 unconstrained endpoint.\n flags[3]\n'
        result=module.analyze(net,report)
        self.assertTrue(result['clean'])
        self.assertEqual(result['constant_endpoints'][0]['tie_cell'],'tie')
        self.assertFalse(module.analyze(net,report.replace('flags[3]','flags[2]'))['clean'])
        self.assertFalse(module.analyze(net,report.replace('flags[3]','alias'))['clean'])

    def test_constant_mux_selects_tied_input_in_kept_enable_register(self):
        net='''module top(input clock,data);
        LOGIC0_X1 lo (.Z(zero)); LOGIC1_X1 hi (.Z(one));
        MUX2_X1 enable (.A(q), .B(zero), .S(one), .Z(d));
        DFF_X1 ff (.D(d), .CK(clock), .Q(q));
        endmodule'''.replace('; ', ';\n')
        report='Warning: There is 1 unconstrained endpoint.\n ff/D\n'
        result=module.analyze(net,report)
        self.assertTrue(result['clean'])
        self.assertEqual(result['constant_mux_proofs']['d']['selector_value'],1)
        self.assertFalse(module.analyze(net.replace('.S(one)','.S(data)'),report)['clean'])
        self.assertFalse(module.analyze(net.replace('.S(one)','.S(zero)'),report)['clean'])
        self.assertFalse(module.analyze(net.replace('.B(zero)','.B(data)'),report)['clean'])
