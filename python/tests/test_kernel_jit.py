import unittest
from qcor import *
# Import Python math with alias
import math as myMathMod

# Some global variables for testing
MY_PI = 3.1416
class TestKernelJIT(unittest.TestCase):
    def test_rewrite_decompose(self):
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

    #     @qjit
    #     def ansatz(q : qreg, x : float):
    #         X(q[0])
    #         with decompose(q, kak) as u:
    #             from scipy.linalg import expm
    #             x0 = np.kron(np.array([[0, 1],[1, 0]]), np.eye(2))
    #             x1 = np.kron(np.eye(2), np.array([[0, 1],[1, 0]]))
    #             y0 = np.kron(np.array([[0, -1j],[1j, 0]]), np.eye(2))
    #             y1 = np.kron(np.eye(2), np.array([[0, -1j],[1j, 0]]))
    #             u = expm(1j * x * (x0 * y1 - x1 * y0))

    #     print(ansatz.get_internal_src())
    #     print(ansatz.get_syntax_handler_src())
    #     q = qalloc(2)
    #     #ansatz(q, 2.2)
    #     #q.print()


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

if __name__ == '__main__':
  unittest.main()