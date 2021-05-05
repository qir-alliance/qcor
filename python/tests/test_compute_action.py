import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
  def test_compute_action(self):
    @qjit
    def x_gate_standalone(q: qubit):
      X(q)

    @qjit
    def test_compute_action1(q : qreg, x : float):
      with compute:
        X.ctrl(q[0], q[1])
        for i in range(3):
            H(q[i+1])
      with action:
        Rz(q[3], x)

    @qjit
    def test_compute_action2(q : qreg, x : float):
      with compute:
        x_gate_standalone.ctrl(q[0], q[1])
        for i in range(3):
            H(q[i+1])
      with action:
        Rz(q[3], x)

    @qjit
    def test_compute_action3(q : qreg, x : float):
      # Control of a kernel compute-action
      test_compute_action1.ctrl(q[0], q[1:q.size()], x)
    
    q = qalloc(5)
    comp0 = test_compute_action1.extract_composite(q, 1.2345)    
    print(comp0)
    # First and last instructions are CNOT's
    self.assertEqual(comp0.getInstruction(0).name(), "CNOT")  
    self.assertEqual(comp0.getInstruction(comp0.nInstructions() - 1).name(), "CNOT") 


    comp1 = test_compute_action2.extract_composite(q, 1.2345)    
    print(comp1)
    # First and last instructions are CNOT's
    self.assertEqual(comp1.getInstruction(0).name(), "CNOT")  
    self.assertEqual(comp1.getInstruction(comp0.nInstructions() - 1).name(), "CNOT") 
    self.assertEqual(comp1.nInstructions(), comp0.nInstructions()) 

    qq = qalloc(6)
    comp2 = test_compute_action3.extract_composite(qq, 1.2345)    
    # only Rz ==> CRz
    self.assertEqual(comp2.nInstructions(), comp0.nInstructions()) 

if __name__ == '__main__':
  unittest.main()