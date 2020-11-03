import unittest
from qcor import *
# Import Python math with alias
import math as myMathMod

# Some global variables for testing
MY_PI = 3.1416
class TestSimpleKernelJIT(unittest.TestCase):
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

    def test_for_loop(self):
        @qjit
        def kernels_w_loops(q : qreg, thetas : List[float], betas : List[float]):
            for i in range(len(q)):
                for theta in thetas:
                    for beta in betas:
                        angle = theta + beta
                        Rx(q[i], angle)
            for i in range(q.size()):
                Measure(q[i])

        list1 = [1.0, 2.0, 3.0]
        list2 = [4.0, 5.0, 6.0]
        q = qalloc(2)
        comp = kernels_w_loops.extract_composite(q, list1, list2)
        self.assertEqual(comp.nInstructions(), q.size() * len(list1) * len(list2) + q.size())
        for i in range(0, q.size() * len(list1) * len(list2)):
            self.assertEqual(comp.getInstruction(i).name(), "Rx") 
        for i in range(q.size() * len(list1) * len(list2), comp.nInstructions()):
            self.assertEqual(comp.getInstruction(i).name(), "Measure") 

    def test_iqft_kernel(self):
        @qjit
        def inverse_qft(q : qreg, startIdx : int, nbQubits : int):
            for i in range(nbQubits/2):
                Swap(q[startIdx + i], q[startIdx + nbQubits - i - 1])
            
            for i in range(nbQubits-1):
                H(q[startIdx+i])
                j = i +1
                for y in range(i, -1, -1):
                    theta = -MY_PI / 2**(j-y)
                    CPhase(q[startIdx+j], q[startIdx + y], theta)
            
            H(q[startIdx+nbQubits-1])
        
        q = qalloc(5)
        comp = inverse_qft.extract_composite(q, 0, 5)
        print(comp.toString())
        self.assertEqual(comp.nInstructions(), 17)   
        self.assertEqual(comp.getInstruction(0).name(), "Swap") 
        self.assertEqual(comp.getInstruction(1).name(), "Swap") 
        self.assertEqual(comp.getInstruction(2).name(), "H") 
        self.assertEqual(comp.getInstruction(3).name(), "CPhase") 
        self.assertEqual(comp.getInstruction(4).name(), "H") 
        for i in range(5, 7):
            self.assertEqual(comp.getInstruction(i).name(), "CPhase") 
        self.assertEqual(comp.getInstruction(7).name(), "H") 
        for i in range(8, 11):
            self.assertEqual(comp.getInstruction(i).name(), "CPhase") 
        self.assertEqual(comp.getInstruction(11).name(), "H") 
        for i in range(12, 16):
            self.assertEqual(comp.getInstruction(i).name(), "CPhase")
        self.assertEqual(comp.getInstruction(16).name(), "H") 
        
    def test_ctrl_kernel(self):
        
        set_qpu('qpp', {'shots':1024})

        @qjit
        def iqft(q : qreg, startIdx : int, nbQubits : int):
            for i in range(nbQubits/2):
                Swap(q[startIdx + i], q[startIdx + nbQubits - i - 1])
            
            for i in range(nbQubits-1):
                H(q[startIdx+i])
                j = i +1
                for y in range(i, -1, -1):
                    theta = -MY_PI / 2**(j-y)
                    CPhase(q[startIdx+j], q[startIdx + y], theta)
            
            H(q[startIdx+nbQubits-1])

        @qjit
        def oracle(q : qreg):
            bit = q.size()-1
            T(q[bit])

        @qjit
        def qpe(q : qreg):
            nq = q.size()
            X(q[nq - 1])
            for i in range(q.size()-1):
                H(q[i])
            
            bitPrecision = nq-1
            for i in range(bitPrecision):
                nbCalls = 2**i
                for j in range(nbCalls):
                    ctrl_bit = i
                    oracle.ctrl(ctrl_bit, q)
            
            # Inverse QFT on the counting qubits
            iqft(q, 0, bitPrecision)
            
            for i in range(bitPrecision):
                Measure(q[i])
        
        q = qalloc(4)
        qpe(q)
        print(q.counts())
        self.assertEqual(q.counts()['100'], 1024)

    def test_adjoint_kernel(self):
        @qjit
        def test_kernel(q : qreg):
            CX(q[0], q[1])
            Rx(q[0], 1.234)
            T(q[0])
            X(q[0])

        @qjit
        def check_adjoint(q : qreg):
            test_kernel.adjoint(q)
        
        q = qalloc(2)
        comp = check_adjoint.extract_composite(q)
        print(comp.toString())
        self.assertEqual(comp.nInstructions(), 4)   
        # Reverse
        self.assertEqual(comp.getInstruction(0).name(), "X") 
        # Check T -> Tdg
        self.assertEqual(comp.getInstruction(1).name(), "Tdg") 
        self.assertEqual(comp.getInstruction(2).name(), "Rx") 
        # Check angle -> -angle
        self.assertAlmostEqual((float)(comp.getInstruction(2).getParameter(0)), -1.234)
        self.assertEqual(comp.getInstruction(3).name(), "CNOT") 


if __name__ == '__main__':
  unittest.main()