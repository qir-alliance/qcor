import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestQCORKernelAutoGrad(unittest.TestCase):
    def test_simple_deuteron(self):
        @qjit
        def ansatz1(q: qreg, theta: List[float]):
            X(q[0])
            Ry(q[1], theta[0])
            CX(q[1], q[0])

        q = qalloc(2)
        H = -2.1433 * X(0) * X(1) - 2.1433 * \
            Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
        
        def objective_function(x):
          q = qalloc(2)
          return  ansatz1.autograd(H, q, x)

        optimizer = createOptimizer('nlopt', {'algorithm':'l-bfgs'})
        (energy, opt_params) = optimizer.optimize(objective_function, 1)
        print("Energy =", energy)
        print("Opt params =", opt_params)
        self.assertAlmostEqual(energy, -1.74886, places=1)

if __name__ == '__main__':
    unittest.main()
