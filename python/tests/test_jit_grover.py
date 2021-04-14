import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
    def test_grover(self):
        set_qpu('qpp', {'shots':1024})
        
        @qjit
        def oracle_fn(q: qreg):
            CZ(q[0], q[2])
            CZ(q[1], q[2])
        
        @qjit
        def reflect_about_uniform(q: qreg):
            with compute:
                H(q)
                X(q)
            with action:
                Z.ctrl(q[0: q.size() - 1], q[q.size() - 1])
            
        @qjit
        def run_grover(q: qreg, oracle_var: KernelSignature(qreg), iterations: int):
            H(q)
            #Iteratively apply the oracle then reflect
            for i in range(iterations):
                oracle_var(q)
                reflect_about_uniform(q)
            # Measure all qubits
            Measure(q)

        q = qalloc(3)
        run_grover(q, oracle_fn, 1)
        q.print()
        counts = q.counts()
        print(counts)
        # Only 2 bitstrings
        self.assertEqual(len(counts), 2)

if __name__ == '__main__':
  unittest.main()