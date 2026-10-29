import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pvt import matching_points


class OperatingPointTest(unittest.TestCase):
    def test_active_operating_condition_takes_precedence_over_nominal_metadata(self):
        with tempfile.TemporaryDirectory()as folder:
            std=Path(folder)/'std.lib';macro=Path(folder)/'sram.lib'
            nominal='nom_process : 1; nom_voltage : 0.7; nom_temperature : 25;'
            std.write_text(nominal)
            macro.write_text(nominal+' default_operating_conditions : TT; operating_conditions (TT) { process : 1; voltage : 0.7; temperature : 0; }')
            self.assertFalse(matching_points(std,macro)['matched'])

    def test_same_corner_name_does_not_hide_a_temperature_mismatch(self):
        with tempfile.TemporaryDirectory()as folder:
            std=Path(folder)/'TT_std.lib';macro=Path(folder)/'TT_sram.lib'
            std.write_text('nom_process : 1; nom_voltage : 0.7; nom_temperature : 25;')
            macro.write_text('nom_process : 1; nom_voltage : 0.7; nom_temperature : 0;')
            self.assertFalse(matching_points(std,macro)['matched'])
            macro.write_text('nom_process : 1.0; nom_voltage : 0.70; nom_temperature : 25.0;')
            self.assertTrue(matching_points(std,macro)['matched'])

    def test_missing_nominal_conditions_do_not_certify_a_corner(self):
        with tempfile.TemporaryDirectory()as folder:
            std=Path(folder)/'std.lib';macro=Path(folder)/'sram.lib'
            std.write_text('nom_process : 1; nom_voltage : 0.7; nom_temperature : 25;')
            macro.write_text('nom_process : 1; nom_voltage : 0.7;')
            self.assertFalse(matching_points(std,macro)['matched'])


if __name__=='__main__':unittest.main()
