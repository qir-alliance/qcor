import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
  def test_compute_action(self):
    @qjit
    def test_compute_action1(q : qreg, x : float):
      with compute:
        X.ctrl(q[0], q[1])
        for i in range(3):
            H(q[i+1])
      with action:
        Rz(q[3], x)

    q = qalloc(5)
    comp0 = test_compute_action1.extract_composite(q, 1.2345)    
    print(comp0)
    # First and last instructions are CNOT's
    self.assertEqual(comp0.getInstruction(0).name(), "CNOT")  
    self.assertEqual(comp0.getInstruction(comp0.nInstructions() - 1).name(), "CNOT") 

if __name__ == '__main__':
  unittest.main()