import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
    def test_opt_level(self):
        @qjit
        def test1(q : qreg):
            H(q[0])
            X(q[0])
            H(q[0])

        q = qalloc(1)
        comp = test1.extract_composite(q)
        self.assertEqual(comp.nInstructions(), 3)   
        # Activate opt-level 1
        set_opt_level(1)
        comp_opt = test1.extract_composite(q)
        # Should become 1 gate: Z, i.e. Rz(pi)
        self.assertEqual(comp_opt.nInstructions(), 1)   
    
    def test_placement_query(self):
        placement_names = get_placement_names()
        self.assertTrue(len(placement_names) > 0)   
        print(placement_names)

    def test_placement(self):
        @qjit
        def test2(q : qreg):
            CX(q[0], q[3])
        # Define a connectivity topology
        set_qpu('qpp', { 'connectivity': [[0,1], [0,2], [1,2], [2,3], [2,4], [3,4]] })
        set_placement('swap-shortest-path')
        q = qalloc(5)
        comp = test2.extract_composite(q)
        print(comp)
        # This becomes 1 SWAP + 1 CNOT => 4 CNOT's
        self.assertEqual(comp.nInstructions(), 4)    

if __name__ == '__main__':
  unittest.main()