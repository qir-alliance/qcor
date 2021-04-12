import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
    def test_multiple_control_kernel(self):
        set_qpu('qpp', {'shots':1024})

        @qjit
        def test_cccx_gate(q : qreg):
            for i in range(q.size()):
                X(q[i])
            # 3 control bits
            X.ctrl([q[1], q[2], q[3]], q[0])
            for i in range(q.size()):
                Measure(q[i])
        
        q = qalloc(4)
        comp = test_cccx_gate.extract_composite(q)
        print(comp)

        # Run experiment
        test_cccx_gate(q)

        # Print the results
        q.print()
        counts = q.counts()
        print(counts)
        self.assertEqual(len(counts), 1)
        # q0: 1 --> 0
        self.assertTrue('0111' in counts)

if __name__ == '__main__':
  unittest.main()