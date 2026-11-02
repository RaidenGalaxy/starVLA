import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from observable_cone import certify


class ObservableConeTest(unittest.TestCase):
    def fixture(self):
        def ff(d,q):return dict(type='DFF',parameters={},port_directions={'D':'input','Q':'output','CLK':'input'},connections={'D':[d],'Q':[q],'CLK':[2]})
        module=dict(ports={'clock':{'direction':'input','bits':[2]},'d':{'direction':'input','bits':[3]},'q':{'direction':'output','bits':[4]}},cells={'live':ff(3,4),'unused_state':ff(5,5)},netnames={'state':{'bits':[4],'attributes':{'init':'0'}}})
        old={'modules':{'Probe':module}};new=copy.deepcopy(old);del new['modules']['Probe']['cells']['unused_state']
        return old,new
    def test_dead_sequential_feedback_is_removed_and_live_state_retained(self):
        old,new=self.fixture();self.assertTrue(certify(old,new,'Probe')['passed'])
    def test_reachable_data_rewire_is_rejected(self):
        old,new=self.fixture();new['modules']['Probe']['cells']['live']['connections']['D']=[2]
        with self.assertRaises(ValueError):certify(old,new,'Probe')
    def test_reachable_flop_removal_is_rejected(self):
        old,new=self.fixture();new['modules']['Probe']['cells'].clear()
        with self.assertRaises(ValueError):certify(old,new,'Probe')
    def test_changed_initial_state_is_rejected(self):
        old,new=self.fixture();new['modules']['Probe']['netnames']['state']['attributes']['init']='1'
        with self.assertRaises(ValueError):certify(old,new,'Probe')
