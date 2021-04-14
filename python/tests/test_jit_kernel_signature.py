import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
    def test_kernel_signature(self):
        set_qpu('qpp', {'shots':1024})
        
        @qjit
        def test_kernel(q: qreg, call_var1: KernelSignature(qreg, int, float), call_var2: KernelSignature(qreg, int, float)):
            call_var1(q, 0, 1.0)
            call_var1(q, 1, 2.0)
            call_var2(q, 0, 1.0)
            call_var2(q, 1, 2.0)

        # These kernels are unknown to test_kernel 
        @qjit
        def rx_kernel(q: qreg, idx: int, theta: float):
            Rx(q[idx], theta)

        @qjit
        def ry_kernel(q: qreg, idx: int, theta: float):
            Ry(q[idx], theta)

        q = qalloc(2)
        comp = test_kernel.extract_composite(q, rx_kernel, ry_kernel)
        print(comp)
        self.assertEqual(comp.nInstructions(), 4)   
        counter = 0
        for i in range(2):
            self.assertEqual(comp.getInstruction(counter).name(), "Rx") 
            self.assertAlmostEqual((float)(comp.getInstruction(counter).getParameter(0)), i + 1.0)
            counter+=1
        for i in range(2):
            self.assertEqual(comp.getInstruction(counter).name(), "Ry") 
            self.assertAlmostEqual((float)(comp.getInstruction(counter).getParameter(0)), i + 1.0)
            counter+=1

if __name__ == '__main__':
  unittest.main()