import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
    def test_grover(self):
        set_qpu('qpp', {'shots':1024})
        
        @qjit
        def rx_kernel(q: qreg, idx: int, theta: float):
            Rx(q[idx], theta)
        
        @qjit
        def test_kernel(q: qreg, call_var: KernelSignature(qreg, int, float)):
            call_var(q, 0, 1.0)
            call_var(q, 1, 2.0)
            # TODO: currently, we don't have the ability to inject
            # new dependency, hence must use rx_kernel here to 
            # pull rx_kernel in.
            rx_kernel(q, 2, 3.0)

        q = qalloc(3)
        test_kernel(q, rx_kernel)
        comp = test_kernel.extract_composite(q, rx_kernel)
        print(comp)
        self.assertEqual(comp.nInstructions(), 3)   
        for i in range(3):
            self.assertEqual(comp.getInstruction(i).name(), "Rx") 
            self.assertAlmostEqual((float)(comp.getInstruction(i).getParameter(0)), i + 1.0)

if __name__ == '__main__':
  unittest.main()