import faulthandler
faulthandler.enable()

import unittest
from qcor import *
# Import Python math with alias
import math as myMathMod

# Some global variables for testing
MY_PI = 3.1416
class TestKernelJIT(unittest.TestCase):
    def test_as_unitary(self):
        try:
           import numpy as np
        except:
            print('No numpy, cant run test_as_unitary')
            return
        
        @qjit
        def dansatz(q : qreg, x : float):
            X(q[0])
            Ry(q[1], x)
            CX(q[1], q[0])
        
        u_mat = dansatz.as_unitary_matrix(qalloc(2), .59)

        H = -2.1433 * X(0) * X(1) - 2.1433 * \
            Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
        Hmat = H.to_numpy()

        # Compute |psi> = U |0>
        zero_state = np.array([1., 0., 0., 0.])
        final_state = np.dot(u_mat, np.transpose(zero_state))
        # Compute E = <psi| H |psi>
        energy = np.dot(final_state, np.dot(Hmat,final_state))
        print(energy)
        self.assertAlmostEqual(energy, -1.74, places=1)

    def test_rewrite_decompose(self):
        try:
           import numpy as np
        except:
            print('No numpy, cant run test_rewrite_decompose')
            return

        set_qpu('qpp', {'shots':1024})
        @qjit
        def foo(q : qreg):
            for i in range(q.size()):
                X(q[i])
            
            with decompose(q) as ccnot:
                ccnot = np.eye(8)
                ccnot[6,6] = 0.0
                ccnot[7,7] = 0.0
                ccnot[6,7] = 1.0
                ccnot[7,6] = 1.0
            
            for i in range(q.size()):
                Measure(q[i])

        print(foo.src)
        q = qalloc(3)
        foo(q)
        counts = q.counts()
        self.assertTrue('110' in counts)
        self.assertTrue(counts['110'] == 1024)

        @qjit
        def all_x(q : qreg):
            with decompose(q) as x_kron:
                sx = np.array([[0, 1],[1, 0]])
                x_kron = np.kron(np.kron(sx,sx),sx)
            
            for i in range(q.size()):
                Measure(q[i])

        print(all_x.src)
        q = qalloc(3)
        all_x(q)
        counts = q.counts()
        print(counts)
        self.assertTrue('111' in counts)
        self.assertTrue(counts['111'] == 1024)

        @qjit
        def try_two_decompose(q : qreg):
            for i in range(q.size()):
                X(q[i])
            
            with decompose(q) as ccnot:
                ccnot = numpy.eye(8)
                ccnot[6,6] = 0.0
                ccnot[7,7] = 0.0
                ccnot[6,7] = 1.0
                ccnot[7,6] = 1.0

            # should have 110
            with decompose(q) as x_kron:
                sx = np.array([[0, 1],[1, 0]])
                x_kron = np.kron(np.kron(sx,sx),sx)
            
            # Should have flipped the bits
            for i in range(q.size()):
                Measure(q[i])

        print(try_two_decompose.src)
        q = qalloc(3)
        try_two_decompose(q)
        counts = q.counts()
        print(counts)
        self.assertTrue('001' in counts)
        self.assertTrue(counts['001'] == 1024)
    
    def test_more_decompose(self):
        try:
           import numpy as np
        except:
            print('No numpy, cant run test_more_decompose')
            return

        set_qpu('qpp', {'shots':1024})
       
        @qjit
        def random_2qbit(q : qreg):
            with decompose(q, kak) as random_unitary:
                a = np.random.rand(4,4)
                qm, r = np.linalg.qr(a, mode='complete')
                random_unitary = qm
            
            for i in range(q.size()):
                Measure(q[i])
        
        print(random_2qbit.src)
        q = qalloc(2)
        print(random_2qbit.extract_composite(q).toString())
        
        @qjit
        def random_1qbit(q : qreg):
            with decompose(q, z_y_z) as random_unitary:
                random_unitary, _ = np.linalg.qr(np.random.rand(2,2), mode='complete')
            
            for i in range(q.size()):
                Measure(q[i])
        
        print(random_1qbit.get_internal_src())
        q = qalloc(2)
        print(random_1qbit.get_syntax_handler_src())
        print(random_1qbit.extract_composite(q).toString())

    # def test_decompose_param(self):
    #     set_qpu('qpp')
    #     @qjit
    #     def ansatz(q : qreg, x : List[float]):
    #         X(q[0])
    #         with decompose(q, kak) as u:
    #             from scipy.sparse.linalg import expm
    #             from openfermion.ops import QubitOperator
    #             from openfermion.transforms import get_sparse_operator
    #             qop = QubitOperator('X0 Y1') - QubitOperator('Y0 X1')
    #             qubit_sparse = get_sparse_operator(qop)
    #             u = expm(0.5j * x[0] * qubit_sparse).todense()

    #     H = -2.1433 * X(0) * X(1) - 2.1433 * \
    #         Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
    #     o = createObjectiveFunction(ansatz, H, 1, {'gradient-strategy':'central', 'step':1e-1})
    #     opt = createOptimizer('nlopt', {'initial-parameters':[.5], 'maxeval':10, 'algorithm':'l-bfgs'})
    #     results = opt.optimize(o)
    #     print("WE ARE DONE")
    #     self.assertAlmostEqual(results[0], -1.74, places=1)

if __name__ == '__main__':
  unittest.main()