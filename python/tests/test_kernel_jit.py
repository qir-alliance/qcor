import unittest
from qcor import *
# Import Python math with alias
import math as myMathMod

# Some global variables for testing
MY_PI = 3.1416
class TestSimpleKernelJIT(unittest.TestCase):
    def test_simple_bell(self):

        set_qpu('qpp', {'shots':8192})

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

        set_qpu('qpp', {'shots':8192})

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

        set_qpu('qpp', {'shots':8192})

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
        self.assertTrue(counts['11'] > 8000)
    
    def test_ModuleConstants(self):

        set_qpu('qpp', {'shots':8192})

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
    
    def test_multiple_kernels(self):
        @qjit
        def apply_H(q : qreg):
            for i in range(q.size()):
                H(q[i])
        
        @qjit
        def apply_Rx(q : qreg, theta: float):
            for i in range(q.size()):
                Rx(q[i], theta)

        @qjit
        def measure_all(q : qreg):
            for i in range(q.size()):
                Measure(q[i])

        @qjit
        def entry_kernel(q : qreg, theta: float):
           apply_H(q)
           apply_Rx(q, theta)
           measure_all(q) 
        
        q = qalloc(5)
        angle = 1.234
        comp = entry_kernel.extract_composite(q, angle)
        self.assertEqual(comp.nInstructions(), 15)   
        for i in range(5):
            self.assertEqual(comp.getInstruction(i).name(), "H") 
        for i in range(5, 10):
            self.assertEqual(comp.getInstruction(i).name(), "Rx") 
            self.assertAlmostEqual((float)(comp.getInstruction(i).getParameter(0)), angle)
        for i in range(10, 15):
            self.assertEqual(comp.getInstruction(i).name(), "Measure") 

    # Make sure that multi-level dependency can be resolved.
    def test_nested_kernels(self):
        @qjit
        def apply_cnot_fwd(q : qreg):
            for i in range(q.size() - 1):
                CX(q[i], q[i + 1])
        
        @qjit
        def make_bell(q : qreg):
            H(q[0])
            apply_cnot_fwd(q)

        @qjit
        def measure_all_bits(q : qreg):
            for i in range(q.size()):
                Measure(q[i])

        @qjit
        def bell_expr(q : qreg):
           # dep: apply_cnot_fwd -> make_bell -> bell_expr
           make_bell(q)
           measure_all_bits(q) 
        
        q = qalloc(5)
        comp = bell_expr.extract_composite(q)
        # 1 H, 4 CNOT, 5 Measure
        self.assertEqual(comp.nInstructions(), 1 + 4 + 5)   
        self.assertEqual(comp.getInstruction(0).name(), "H") 
        for i in range(1, 5):
            self.assertEqual(comp.getInstruction(i).name(), "CNOT") 
        for i in range(5, 10):
            self.assertEqual(comp.getInstruction(i).name(), "Measure") 

if __name__ == '__main__':
  unittest.main()