import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
  def test_simple(self):
    @qjit
    def x_gate_standalone(q: qubit):
      X(q)

    @qjit
    def test_compute_action1(q : qreg, x : float):
      with compute:
        # control of x gate
        X.ctrl(q[0], q[1])
        for i in range(3):
            H(q[i+1])
      with action:
        Rz(q[3], x)

    @qjit
    def test_compute_action2(q : qreg, x : float):
      with compute:
        # control of x gate wrapped in a kernel
        x_gate_standalone.ctrl(q[0], q[1])
        for i in range(3):
            H(q[i+1])
      with action:
        Rz(q[3], x)

    @qjit
    def test_compute_action3(q : qreg, x : float):
      # Control of a kernel compute-action (version 1: ctrl of simple gate)
      test_compute_action1.ctrl(q[0], q[1:q.size()], x)

    @qjit
    def test_compute_action4(q : qreg, x : float):
      # Control of a kernel compute-action (version 2: ctrl of a kernel)
      test_compute_action2.ctrl(q[0], q[1:q.size()], x)
    
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
    print(comp2)
    # First and last instructions are CNOT's
    self.assertEqual(comp2.getInstruction(0).name(), "CNOT")  
    self.assertEqual(comp2.getInstruction(comp0.nInstructions() - 1).name(), "CNOT") 
    middle_inst_idx = (int) (comp2.nInstructions() / 2)
    # only Rz ==> CRz
    self.assertEqual(comp2.getInstruction(middle_inst_idx).name(), 'CRZ')
    self.assertEqual(comp2.nInstructions(), comp0.nInstructions()) 

    comp3 = test_compute_action4.extract_composite(qq, 1.2345)    
    print(comp3)
    # First and last instructions are CNOT's
    self.assertEqual(comp3.getInstruction(0).name(), "CNOT")  
    self.assertEqual(comp3.getInstruction(comp0.nInstructions() - 1).name(), "CNOT") 
    middle_inst_idx = (int) (comp3.nInstructions() / 2)
    # only Rz ==> CRz
    self.assertEqual(comp3.getInstruction(middle_inst_idx).name(), 'CRZ')
    self.assertEqual(comp3.nInstructions(), comp0.nInstructions()) 

  def test_multi_control(self):
    @qjit
    def x_gate_standalone_m(q: qubit):
      X(q)

    @qjit
    def test_compute_action1_m(q : qreg, x : float):
      with compute:
        # CCX
        X.ctrl(q[0:2], q[2])
      with action:
        Rz(q[3], x)

    @qjit
    def test_compute_action2_m(q : qreg, x : float):
      with compute:
        # control of x gate wrapped in a kernel
        x_gate_standalone_m.ctrl(q[0:2], q[2])
      with action:
        Rz(q[3], x)

    @qjit
    def test_compute_action3_m(q : qreg, x : float):
      # Control of a kernel compute-action (version 1: ctrl of simple gate)
      test_compute_action1_m.ctrl(q[0], q[1:q.size()], x)

    @qjit
    def test_compute_action4_m(q : qreg, x : float):
      # Control of a kernel compute-action (version 2: ctrl of a kernel)
      test_compute_action2_m.ctrl(q[0], q[1:q.size()], x)
    
    q = qalloc(5)
    comp0 = test_compute_action1_m.extract_composite(q, 1.2345)    
    print(comp0)
    # 2 CCX and 1 Rz
    self.assertEqual(comp0.nInstructions(), 31)
    comp1 = test_compute_action2_m.extract_composite(q, 1.2345)    
    self.assertEqual(comp1.nInstructions(), 31)

    comp2 = test_compute_action3_m.extract_composite(q, 1.2345)    
    self.assertEqual(comp2.nInstructions(), 31)
    comp3 = test_compute_action4_m.extract_composite(q, 1.2345)    
    self.assertEqual(comp3.nInstructions(), 31)
    # Check outer loop control
    self.assertEqual(comp0.getInstruction(15).name(), 'Rz')
    self.assertEqual(comp1.getInstruction(15).name(), 'Rz')
    self.assertEqual(comp2.getInstruction(15).name(), 'CRZ')
    self.assertEqual(comp3.getInstruction(15).name(), 'CRZ')

if __name__ == '__main__':
  unittest.main()