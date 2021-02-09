import unittest
from qcor import *

class TestKernelBuilder(unittest.TestCase):
    def test_simple_bell(self):
        set_qpu('qpp', {'shots':100})
        builder = KernelBuilder()
        builder.h(0)
        for i in range(5):
            builder.cnot(i, i+1)
        builder.measure_all()
        ghz_6 = builder.create()
        q = qalloc(6)
        ghz_6(q)
        counts = q.counts()
        print(counts)
        self.assertTrue('111111' in counts)
        self.assertTrue('000000' in counts)
        self.assertTrue(len(counts) == 2)
    
    def test_variational_spec(self):
        set_qpu('qpp')
        builder = KernelBuilder()

        builder.x(0)
        builder.ry(1, 't')
        builder.cnot(1, 0) 

        ansatz = builder.create()

        H = -2.1433 * X(0) * X(1) - 2.1433 * \
            Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

        n_params = 1
        obj = createObjectiveFunction(ansatz, H, n_params)

        # evaluate at a concrete set of params
        vqe_energy = obj([.59])
        self.assertAlmostEqual(vqe_energy, -1.74, places=1)


        # Run full optimization
        optimizer = createOptimizer('nlopt')
        results = optimizer.optimize(obj)
        self.assertAlmostEqual(results[0], -1.74, places=1)
    
    def test_from_qasm(self):
        set_qpu('qpp', {'shots':100})
        src = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
'''
        builder = KernelBuilder()
        builder.from_qasm(src)
        test_from_qasm = builder.create()

        q = qalloc(3)
        test_from_qasm(q)
        counts = q.counts()
        self.assertTrue('111' in counts)
        self.assertTrue('000' in counts)
        self.assertTrue(len(counts) == 2)

    
    # def test_synthesis(self):
    def test_exp(self):
        exponent_op = adag(0) * a(1) - adag(1) * a(0)

        builder = KernelBuilder()
        builder.x(0)
        builder.exp('theta', exponent_op)
        ansatz = builder.create()
        # Define the hamiltonian
        H = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907
        # Create the ObjectiveFunction, specify central finite diff gradient
        obj = createObjectiveFunction(ansatz, H, 1, {'gradient-strategy':'central', 'step':1e-1})
        # create the lbfgs optimizer
        optimizer = createOptimizer('nlopt', {'algorithm':'l-bfgs', 'ftol':1e-3})
        results = optimizer.optimize(obj)
        self.assertAlmostEqual(results[0], -1.74, places=1)

    def test_synthesis(self):
        set_qpu('qpp', {'shots':100})
        try:
           import numpy as np
        except:
            print('No numpy, cant run test_synthesis')
            return

        ccnot = np.eye(8)
        ccnot[6,6] = 0.0
        ccnot[7,7] = 0.0
        ccnot[6,7] = 1.0
        ccnot[7,6] = 1.0

        # Synthesize the CCNOT kernel 
        # using the KernelBuilder
        builder = KernelBuilder()
        [builder.x(i) for i in range(3)]
        builder.synthesize(unitary=ccnot)
        builder.measure(range(3))
        ccnot_circuit = builder.create()    

        q = qalloc(3)
        ccnot_circuit(q)
        counts = q.counts()
        self.assertTrue(len(counts) == 1)
        self.assertTrue('110' in counts)

    def test_call_previous_kernels(self):
        builder = KernelBuilder()

        builder.x(0)
        builder.ry(1, 't')
        builder.cnot(1, 0) 
        ansatz = builder.create()

        @qjit
        def x0x1(q : qreg, t:float):
            ansatz(q, t)
            H(q[0])
            H(q[1])
            Measure(q[0])
            Measure(q[1])

        q = qalloc(2)
        x0x1(q, 2.2)
        self.assertAlmostEqual(q.exp_val_z(), .8085, places=1)

        b = KernelBuilder()
        b.invoke(ansatz)
        b.h(0)
        b.h(1)
        b.measure_all()
        test = b.create()

        q = qalloc(2)
        test(q, 2.2)
        self.assertAlmostEqual(q.exp_val_z(), .8085, places=1)







if __name__ == '__main__':
  unittest.main()