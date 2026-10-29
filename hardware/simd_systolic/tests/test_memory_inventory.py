import importlib.util,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('mapped_memory_inventory',ROOT/'scripts/memory_inventory.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RetainedMemoryTest(unittest.TestCase):
    def test_declared_or_constant_words_do_not_count_as_retained_storage(self):
        with tempfile.TemporaryDirectory()as temporary:
            net=Path(temporary)/'synth.v'
            net.write_text('''module Top();
wire [127:0] \\tile.a_ext.Memory[1000] ;
assign \\tile.a_ext.Memory[1000] = 128'b0;
DFF_X1 first (.Q(\\tile.a_ext.Memory[0] [0]));
DFF_X1 second (.QN(\\b_ext.Memory[2047] [127]));
DFF_X1 third (.Q(\\tile.aggregates_ext.Memory[8191] [63]));
endmodule
''')
            result=module.inventory(net)
            self.assertEqual(result['retained_bits'],dict(a_ext=1,b_ext=1,c_ext=0,aggregates_ext=1))
            self.assertFalse(result['all_nominal_bits_retained'])
