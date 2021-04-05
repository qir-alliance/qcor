import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
    def test_multiple_control_kernel(self):
        @qjit
        def apply_X_at_idx(q : qreg, idx: int):
            X(q[idx])

        @qjit
        def test_ccx_gate(q : qreg):
           apply_X_at_idx.ctrl([1, 2], q, 0)
           for i in range(q.size()):
                Measure(q[i])
        
        q = qalloc(3)
        comp = test_ccx_gate.extract_composite(q)
        print(comp)
        
if __name__ == '__main__':
  unittest.main()