import unittest
from qcor import *
try:
    from openfermion.ops import FermionOperator as FOp
    from openfermion.ops import QubitOperator as QOp
    from openfermion.transforms import reverse_jordan_wigner, jordan_wigner
    
    class TestOpenFermion(unittest.TestCase):
        def test_simple_fermion(self):
            # Create Operator as OpenFermion FermionOperator
            H = FOp('', 0.0002899) + FOp('0^ 0', -.43658) + \
                FOp('1 0^', 4.2866) + FOp('1^ 0', -4.2866) + FOp('1^ 1', 12.25) 
              
            @qjit
            def ansatz(q: qreg, theta: float):
                X(q[0])
                Ry(q[1], theta)
                CX(q[1], q[0])
        
            n_params = 1
            obj = createObjectiveFunction(ansatz, H, n_params, {'gradient-strategy':'parameter-shift'})
            optimizer = createOptimizer('nlopt', {'nlopt-optimizer':'l-bfgs'})
            results = optimizer.optimize(obj)
            self.assertAlmostEqual(results[0], -1.74, places=1) 

        def test_simple_qubit(self):
            # Create Operator as OpenFermion FermionOperator
            H = QOp('', 5.907) + QOp('Y0 Y1', -2.1433) + \
                QOp('X0 X1', -2.1433) + QOp('Z0', .21829) + QOp('Z1', -6.125) 
              
            @qjit
            def ansatz(q: qreg, theta: float):
                X(q[0])
                Ry(q[1], theta)
                CX(q[1], q[0])
        
            n_params = 1
            obj = createObjectiveFunction(ansatz, H, n_params, {'gradient-strategy':'parameter-shift'})
            optimizer = createOptimizer('nlopt', {'nlopt-optimizer':'l-bfgs'})
            results = optimizer.optimize(obj)
            self.assertAlmostEqual(results[0], -1.74, places=1) 
except:
    pass

if __name__ == '__main__':
  unittest.main()