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

    def test_kernel_signature_ctrl_adj(self):
        set_qpu('qpp', {'shots':1024})
        
        @qjit
        def test_kernel1(q: qreg, call_var1: KernelSignature(qreg, int, float), call_var2: KernelSignature(qubit)):
            call_var1.adjoint(q, 0, 1.0)
            call_var1.adjoint(q, 1, 2.0)
            call_var2.ctrl(q[1], q[0])

        # These kernels are unknown to test_kernel 
        @qjit
        def rz_kernel(q: qreg, idx: int, theta: float):
            Rz(q[idx], theta)

        @qjit
        def x_kernel(q: qubit):
            X(q)

        q = qalloc(2)
        comp = test_kernel1.extract_composite(q, rz_kernel, x_kernel)
        print(comp)
        self.assertEqual(comp.nInstructions(), 3)   
        counter = 0
        for i in range(2):
            self.assertEqual(comp.getInstruction(counter).name(), "Rz") 
            # Minus due to adjoint
            self.assertAlmostEqual((float)(comp.getInstruction(counter).getParameter(0)), -(i + 1.0))
            counter+=1
        self.assertEqual(comp.getInstruction(2).name(), "CNOT")
        self.assertEqual(comp.getInstruction(2).bits()[0], 1)
        self.assertEqual(comp.getInstruction(2).bits()[1], 0)
    
    def test_kernel_signature_substitute(self):
        @qjit
        def htest(q : qreg, sp_var : KernelSignature(qreg)):
            H(q[0])
            psi = q[1:q.size()]
            sp_var(psi) 
        
        @qjit
        def sp(q : qreg):
            X(q)
        
        q = qalloc(3)
        comp = htest.extract_composite(q, sp)
        print(comp)
        self.assertEqual(comp.nInstructions(), 3)
        self.assertEqual(comp.getInstruction(0).name(), "H")
        self.assertEqual(comp.getInstruction(0).bits()[0], 0)
        self.assertEqual(comp.getInstruction(1).name(), "X")
        self.assertEqual(comp.getInstruction(1).bits()[0], 1)
        self.assertEqual(comp.getInstruction(2).name(), "X")
        self.assertEqual(comp.getInstruction(2).bits()[0], 2)

if __name__ == '__main__':
  unittest.main()