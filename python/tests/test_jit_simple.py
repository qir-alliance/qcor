import faulthandler
faulthandler.enable()

import unittest
from qcor import *
# Import Python math with alias
import math as myMathMod

# Some global variables for testing
MY_PI = 3.1416
class TestKernelJIT(unittest.TestCase):
    def test_simple_bell(self):

        set_qpu('qpp', {'shots':1024})

        @qjit
        def bell(q : qreg):
            H(q[0])
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        bell(q)

        # Print the results
        q.print()
        self.assertAlmostEqual(q.exp_val_z(), 1.0, places=1)
        counts = q.counts()
        self.assertEqual(len(counts), 2)
        self.assertTrue('00' in counts)
        self.assertTrue('11' in counts)

    def test_assignment(self):

        set_qpu('qpp', {'shots':1024})

        @qjit
        def varAssignKernel(q : qreg):
            # Simple value assignment
            angle = 3.14/4.0
            Rx(q[0], angle)
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        varAssignKernel(q)

        # Print the results
        q.print()
        counts = q.counts()
        self.assertEqual(len(counts), 2)
        self.assertTrue('00' in counts)
        self.assertTrue('11' in counts)
        # Angle less than Pi/2 => 00 more than 11
        self.assertTrue(counts['00'] > counts['11'])
    
    def test_globalVar(self):

        set_qpu('qpp', {'shots':1024})

        @qjit
        def kernelUseGlobals(q : qreg):
            Rx(q[0], MY_PI)
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        kernelUseGlobals(q)

        # Print the results
        q.print()
        counts = q.counts()
        # Pi pulse -> X gate
        self.assertTrue(counts['11'] > 1000)
    
    def test_ModuleConstants(self):

        set_qpu('qpp', {'shots':1024})

        @qjit
        def kernelUseConstants(q : qreg):
            # Use math.pi constant
            Rx(q[0], myMathMod.pi/2)
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        kernelUseConstants(q)

        # Print the results
        q.print()
        counts = q.counts()
        self.assertEqual(len(counts), 2)
        self.assertTrue('00' in counts)
        self.assertTrue('11' in counts)
    
    def test_exp_i_theta(self):
        @qjit
        def kernelExpVar(q : qreg, theta: float):
            exponent_op = X(0) * Y(1) - Y(0) * X(1)
            exp_i_theta(q, theta, exponent_op)
        
        # Allocate 2 qubits
        q = qalloc(2)
        theta = 1.234
        # Validate exp_i_theta expansion
        comp = kernelExpVar.extract_composite(q, theta)
        self.assertEqual(comp.nInstructions(), 14)

    # Test of edge case where the first statement is a for loop
    def test_for_loop(self):
        @qjit
        def testFor(q : qreg):
            for i in range(q.size()):
                H(q[i])
        
        q = qalloc(5)
        comp = testFor.extract_composite(q)
        self.assertEqual(comp.nInstructions(), 5)   

    def test_for_loop_enumerate(self):
        set_qpu('qpp')
        @qjit
        def ansatz(q: qreg, x: List[float], exp_args: List[Operator]):
            X(q[0])
            for i, exp_arg in enumerate(exp_args):
                exp_i_theta(q, x[i], exp_args[i])

        exp_args = [adag(0) * a(1) - adag(1)*a(0), adag(0)*a(2) - adag(2)*a(0)]
        H = createOperator('5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + 9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2')
        energy = ansatz.observe(H, qalloc(3), [0.7118083109334505, 0.27387413138588135], exp_args)
        self.assertAlmostEqual(energy, -2.044, places=1)

    # Test conditional if..elif..else rewrite
    def test_if_clause(self):
        @qjit
        def test_if_stmt(q : qreg, flag: int):
            H(q[0])
            if flag == 0:
                X(q[0])
            elif flag == 1:
                Y(q[0])
            elif flag == 2:
                Z(q[0])
            else:
                T(q[0])
        
        q = qalloc(2)

        # Examine the circuit QASM with various values of flag
        comp0 = test_if_stmt.extract_composite(q, 0)
        comp1 = test_if_stmt.extract_composite(q, 1)
        comp2 = test_if_stmt.extract_composite(q, 2)
        comp3 = test_if_stmt.extract_composite(q, 3)

        self.assertEqual(comp0.nInstructions(), 2)   
        self.assertEqual(comp1.nInstructions(), 2)   
        self.assertEqual(comp2.nInstructions(), 2)   
        self.assertEqual(comp3.nInstructions(), 2)   

        self.assertEqual(comp0.getInstruction(1).name(), "X") 
        self.assertEqual(comp1.getInstruction(1).name(), "Y") 
        self.assertEqual(comp2.getInstruction(1).name(), "Z") 
        self.assertEqual(comp3.getInstruction(1).name(), "T") 

    def test_instBroadCast(self):
        set_qpu('qpp', {'shots':1024})
        
        @qjit
        def broadCastTest(q : qreg):
            # Simple broadcast
            X(q)
            # broadcast by slice
            Z(q[0:q.size()])
            # Even qubits
            Y(q[0:q.size():2])

        q = qalloc(6)
        comp = broadCastTest.extract_composite(q)
        counter = 0
        for i in range(q.size()):
            self.assertEqual(comp.getInstruction(counter).name(), "X") 
            self.assertEqual(comp.getInstruction(counter).bits()[0], i) 
            counter += 1
        
        for i in range(q.size()):
            self.assertEqual(comp.getInstruction(counter).name(), "Z") 
            self.assertEqual(comp.getInstruction(counter).bits()[0], i) 
            counter += 1
        
        for i in range(0, q.size(), 2):
            self.assertEqual(comp.getInstruction(counter).name(), "Y") 
            self.assertEqual(comp.getInstruction(counter).bits()[0], i) 
            counter += 1    
        
        self.assertEqual(comp.nInstructions(), counter)
    
    def test_multi_qregs(self):
        set_qpu('qpp', {'shots':1024})
        
        @qjit 
        def bell_qubits(q: qubit, r: qubit):
            H(q)
            X.ctrl(q, r)
            Measure(q)
            Measure(r)
        
        q = qalloc(1)
        r = qalloc(1)
        bell_qubits(q[0], r[0])
        q.print()
        r.print()
        self.assertEqual(len(q.counts()), 2)
        self.assertEqual(len(r.counts()), 2)
        # entangled
        self.assertEqual(q.counts()["0"], r.counts()["0"])
        self.assertEqual(q.counts()["1"], r.counts()["1"])
    
    def test_while_loop(self):
        @qjit 
        def while_loop_kernel(q: qubit, x: int, exp_inv: int):
            rev = 0
            while x:
                rev <<= 1
                rev += x & 1
                x >>= 1

            if rev == exp_inv:
                print("Success")
                X(q)

        # Reference: pure python to check
        def reverse_bit(num):
            result = 0
            while num:
                result = (result << 1) + (num & 1)
                num >>= 1
            return result
        
        q = qalloc(1)
        test_val = 123
        expected = reverse_bit(test_val)
        comp0 = while_loop_kernel.extract_composite(q[0], test_val, expected)    
        print("Comp:", comp0)
        # Has X applied.
        self.assertEqual(comp0.nInstructions(), 1) 

if __name__ == '__main__':
  unittest.main()