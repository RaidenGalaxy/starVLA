import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'physical'))
from check_async_mapping import run
from async_mapping_policy import POLICY,validate
from run import sha,map_elaborated,tool
from asap7_platform import profile


class AsyncMappingTest(unittest.TestCase):
    def test_actual_liberty_reset_values_polarities_enable_and_release(self):
        with tempfile.TemporaryDirectory()as directory:
            out=Path(directory)/'proof';run(out)
            proof=json.loads((out/'verification.json').read_text())
            identity=dict(async_mapping_policy=POLICY,async_mapping_sha256=proof['mapping_sha256'],
                async_mapping_proof_file=str(out/'verification.json'),
                async_mapping_proof_sha256=sha(out/'verification.json'),liberty_sha256=proof['liberty_sha256'])
            self.assertEqual(proof['event_checks'],5010)
            self.assertEqual(validate(out,identity,out/'mapped.v')['async_cells'],8)
            with self.assertRaisesRegex(ValueError,'unqualified asynchronous'):
                validate(out,{},out/'mapped.v')
            production=Path(directory)/'production';production.mkdir()
            (production/'build_identity.json').write_text('{}\n')
            subprocess.run([tool('yosys'),'-Q','-T','-p',
                f'read_verilog {out/"source.sv"}; synth -top AsyncProbe -flatten -noabc; write_rtlil {production/"premap.il"}'],
                check=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,timeout=60)
            map_elaborated(production,profile('TT')['liberty'],1,os.environ.copy())
            actual_identity=json.loads((production/'build_identity.json').read_text())
            actual_identity['liberty_sha256']=proof['liberty_sha256']
            self.assertEqual(validate(production,actual_identity)['async_cells'],8)
            actual=production/'simulation.v'
            actual.write_text((production/'synth.v').read_text().replace('module AsyncProbe(', 'module AsyncMapped(',1))
            with(production/'build.log').open('w')as stream:
                subprocess.run(['verilator','--cc','--exe','--build','-j','4','-Wno-fatal',
                    '--top-module','AsyncMappingProbe','--Mdir',str(production/'obj'),
                    str(out/'cells.v'),str(out/'reference.sv'),str(actual),str(out/'top.sv'),
                    str(out/'bench.cpp'),'-o',str(production/'sim')],check=True,
                    stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            result=subprocess.run([str(production/'sim')],check=True,capture_output=True,text=True,timeout=30)
            self.assertIn('ASYNC_MAPPING_CHECKS 5010',result.stdout)
            (out/'Asap7AsyncFlops.v').write_text('// changed mapping\n')
            with self.assertRaisesRegex(ValueError,'snapshot changed'):
                validate(out,identity,out/'mapped.v')
