import unittest, math
from qcor import *

class TestWorkflows(unittest.TestCase):
    def test_td_workflow(self):
      # Time-dependent Hamiltonian: 
      # Returns the Pauli operators at a time point.
      def td_hamiltonian(t):
        Jz = 2 * math.pi * 2.86265 * 1e-3
        epsilon = Jz
        omega = 4.8 * 2 * math.pi * 1e-3
        return -Jz * Z(0) * Z(1)  - Jz * Z(1) * Z(2) + (-epsilon * math.cos(omega * t)) * (X(0) + X(1) + X(2)) 

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

    def test_vqe_ansatz(self): 
      
      H = -2.1433 * X(0) * X(1) - 2.1433 * \
            Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
      
      @qjit
      def ansatz(q : qreg, theta : float):
        X(q[0])
        Ry(q[1], theta)
        CX(q[1], q[0])


      num_qubits = 2
      num_params = 1
      problemModel = qsim.ModelBuilder.createModel(ansatz, H, num_qubits, num_params)
      
      optimizer = createOptimizer('nlopt')
      
      workflow = qsim.getWorkflow('vqe', {'optimizer': optimizer})

      result = workflow.execute(problemModel)

      energy = result['energy']
      print(energy)
      self.assertAlmostEqual(energy, -1.74, places=1)

if __name__ == '__main__':
  unittest.main()