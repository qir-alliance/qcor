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
    
    def test_autograd_args_translate(self):
        set_qpu('qpp')
        @qjit
        def ansatz_args_translate(q: qreg, x: List[float], exp_args: List[Operator]):
            X(q[0])
            for i, exp_arg in enumerate(exp_args):
                exp_i_theta(q, x[i], exp_args[i])

        exp_args = [adag(0) * a(1) - adag(1)*a(0), adag(0)*a(2) - adag(2)*a(0)]
        H = createOperator('5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + 9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2')
        
        # Custom arg_translator in a Pythonic way
        def ansatz_translate(self, q: qreg, x: List[float]):
            ret_dict = {}    
            ret_dict["q"] = q
            ret_dict["x"] = x
            ret_dict["exp_args"] = exp_args
            return ret_dict

        ansatz_args_translate.translate = MethodType(ansatz_translate, qjit)
        
        def objective_function(x):
          q = qalloc(3)
          # Autograd a kernel with a custom args translate.
          return ansatz_args_translate.autograd(H, q, x)
        
        optimizer = createOptimizer('nlopt', {'algorithm':'l-bfgs'})
        (energy, opt_params) = optimizer.optimize(objective_function, 2)
        print("Energy =", energy)
        print("Opt params =", opt_params)
        self.assertAlmostEqual(energy, -2.044, places=1)
        self.assertAlmostEqual(opt_params[0], 0.7118, places=1)
        self.assertAlmostEqual(opt_params[1], 0.2739, places=1)

if __name__ == '__main__':
    unittest.main()
