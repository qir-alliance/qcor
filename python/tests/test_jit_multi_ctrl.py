import faulthandler
faulthandler.enable()

import unittest
from qcor import *

class TestKernelJIT(unittest.TestCase):
    def test_multiple_control_kernel(self):
        set_qpu('qpp', {'shots':1024})

        @qjit
        def test_cccx_gate(q : qreg):
            for i in range(q.size()):
                print('Apply gate at', i)
                X(q[i])
            # 3 control bits
            X.ctrl([q[1], q[2], q[3]], q[0])
            for i in range(q.size()):
                Measure(q[i])
        
        q = qalloc(4)
        comp = test_cccx_gate.extract_composite(q)
        print(comp)

        # Run experiment
        test_cccx_gate(q)

        # Print the results
        q.print()
        counts = q.counts()
        print(counts)
        self.assertEqual(len(counts), 1)
        # q0: 1 --> 0
        self.assertTrue('0111' in counts)

    def test_qreg_head_tail(self):
        set_qpu('qpp', {'shots':1024})

        @qjit
        def test_cccx_qreg(q : qreg):
            # Broadcast
            X(q)
            # 3 control bits
            ctrl_qubits = q.tail(q.size() - 1)
            first_qubit = q.head()
            X.ctrl(ctrl_qubits, first_qubit)
            # # Broadcast
            Measure(q)
        
        q = qalloc(4)
        comp = test_cccx_qreg.extract_composite(q)
        print(comp)

        # Run experiment
        test_cccx_qreg(q)

        # Print the results
        q.print()
        counts = q.counts()
        print(counts)
        self.assertEqual(len(counts), 1)
        # q0: 1 --> 0
        self.assertTrue('0111' in counts)
    
    def test_qreg_slicing(self):
        set_qpu('qpp', {'shots':1024})

        @qjit
        def test_cccx_qreg_slice(q : qreg):
            # Broadcast
            X(q)
            # 3 control bits:
            # q[0], q[1], q[2]
            ctrl_qubits = q[0:3]
            last_qubit = q.tail()
            X.ctrl(ctrl_qubits, last_qubit)
            # Broadcast
            Measure(q)
        
        q = qalloc(4)
        comp = test_cccx_qreg_slice.extract_composite(q)
        print(comp)

        # Run experiment
        test_cccx_qreg_slice(q)

        # Print the results
        q.print()
        counts = q.counts()
        print(counts)
        self.assertEqual(len(counts), 1)
        # q3: 1 --> 0
        self.assertTrue('1110' in counts)
    
    def test_qreg_slicing_inline(self):
        set_qpu('qpp', {'shots':1024})

        @qjit
        def test_cccx_qreg_slice_inline(q : qreg):
            # Broadcast via a slice
            X(q)
            # Control with slicing inline
            X.ctrl(q[0:3], q.tail())
            # Broadcast
            Measure(q)
        
        q = qalloc(4)
        comp = test_cccx_qreg_slice_inline.extract_composite(q)
        print(comp)

        # Run experiment
        test_cccx_qreg_slice_inline(q)

        # Print the results
        q.print()
        counts = q.counts()
        print(counts)
        self.assertEqual(len(counts), 1)
        # q3: 1 --> 0
        self.assertTrue('1110' in counts)

if __name__ == '__main__':
  unittest.main()