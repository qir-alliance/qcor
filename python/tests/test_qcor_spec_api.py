import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestQCORSpecAPI(unittest.TestCase):
    def test_simple_deuteron(self):

        @qjit
        def ansatz(q: qreg, theta: List[float]):
            X(q[0])
            Ry(q[1], theta[0])
            CX(q[1], q[0])

        q = qalloc(2)
        
        comp = ansatz.extract_composite(q, [2.2])
        print(comp.toString())

        H = -2.1433 * X(0) * X(1) - 2.1433 * \
            Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
        
        n_params = 1
        obj = createObjectiveFunction(ansatz, H, n_params)
        vqe_energy = obj([.59])
        self.assertAlmostEqual(vqe_energy, -1.74, places=1)

        optimizer = createOptimizer('nlopt')

        results = optimizer.optimize(obj)

        self.assertAlmostEqual(results[0], -1.74, places=1)
        print(results)

        print(ansatz.openqasm(q, [2.2]))

    def test_simple_deuteron_with_grad(self):

        @qjit
        def ansatz(q: qreg, theta: float):
            X(q[0])
            Ry(q[1], theta)
            CX(q[1], q[0])
        
        H = -2.1433 * X(0) * X(1) - 2.1433 * \
            Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
        
        n_params = 1
        obj = createObjectiveFunction(ansatz, H, n_params, {'gradient-strategy':'parameter-shift'})
        optimizer = createOptimizer('nlopt', {'nlopt-optimizer':'l-bfgs'})
        results = optimizer.optimize(obj)
        self.assertAlmostEqual(results[0], -1.74, places=1)
        print(results)

    def test_observe(self):
        H = -2.1433 * X(0) * X(1) - 2.1433 * \
            Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

        @qjit
        def ansatz(q : qreg, theta : float):
            X(q[0])
            Ry(q[1], theta)
            CX(q[1], q[0])

        target_energy = -1.74

        def objective_function(x):
            q = qalloc(H.nBits())
            energy = ansatz.observe(H, q, x[0])
            return abs(target_energy - energy)

        optimizer = createOptimizer('nlopt', {'nlopt-maxeval':20})
        opt_val, opt_params = optimizer.optimize(objective_function, 1)   
        self.assertAlmostEqual(opt_val, 0.0, places=1)

    def test_observe_openfermion(self):
        try:
            from openfermion.ops import FermionOperator as FOp
            from openfermion.ops import QubitOperator as QOp
            from openfermion.transforms import jordan_wigner

            H = FOp('', 0.0002899) + FOp('0^ 0', -.43658) + \
                FOp('1 0^', 4.2866) + FOp('1^ 0', -4.2866) + FOp('1^ 1', 12.25) 
            
            @qjit
            def ansatz(q : qreg, theta : float):
                X(q[0])
                Ry(q[1], theta)
                CX(q[1], q[0])
            
            target_energy = -1.74

            def objective_function(x):
                q = qalloc(2)
                energy = ansatz.observe(H, q, x[0])
                print(energy)
                return abs(target_energy - energy)

            optimizer = createOptimizer('nlopt', {'nlopt-maxeval':20})
            opt_val, opt_params = optimizer.optimize(objective_function, 1)   
            print(opt_val, opt_params)
            self.assertAlmostEqual(opt_val, 0.0, places=1)
 
            Hq = jordan_wigner(H)
            def objective_function2(x):
                q = qalloc(2)
                energy = ansatz.observe(Hq, q, x[0])
                print(energy)
                return abs(target_energy - energy)

            objective_function2([2.2])
            optimizer = createOptimizer('nlopt', {'nlopt-maxeval':20})
            opt_val, opt_params = optimizer.optimize(objective_function2, 1)   
            self.assertAlmostEqual(opt_val, 0.0, places=1)

        except:
            pass

    def test_operator(self):
        H = createOperator('-2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + 5.907')
        print(H)


if __name__ == '__main__':
    unittest.main()
