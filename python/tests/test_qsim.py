import unittest
from qcor import *
import numpy as np

class TestWorkflows(unittest.TestCase):
    def test_td_workflow(self):
      # Time-dependent Hamiltonian: 
      # Returns the Pauli operators at a time point.
      def td_hamiltonian(t):
        Jz = 2 * np.pi * 2.86265 * 1e-3
        epsilon = Jz
        omega = 4.8 * 2 * np.pi * 1e-3
        return -Jz * Z(0) * Z(1)  - Jz * Z(1) * Z(2) + (-epsilon * np.cos(omega * t)) * (X(0) + X(1) + X(2)) 

      # Observable = average magnetization
      observable = (1.0 / 3.0) * (Z(0) + Z(1) + Z(2))
      problemModel = qsim.ModelBuilder.createModel(observable, td_hamiltonian)
      nbSteps = 100
      workflow = qsim.getWorkflow(
        "td-evolution", {"method": "trotter", "dt": 3.0, "steps": nbSteps})
      result = workflow.execute(problemModel)
      self.assertEqual(len(result["exp-vals"]), nbSteps + 1)
      self.assertAlmostEqual(result["exp-vals"][0], 1.0, places=1)
      self.assertAlmostEqual(result["exp-vals"][nbSteps], 0.5, places=1)

if __name__ == '__main__':
  unittest.main()