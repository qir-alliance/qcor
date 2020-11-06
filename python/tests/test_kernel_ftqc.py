import unittest
from qcor import *

float_result = 0.0
int_result = 0

class TestKernelFTQC(unittest.TestCase):
    def test_pass_by_ref(self):
        @qjit
        def testX0(q : qreg, out_meas_z: FLOAT_REF):
            X(q[0])
            if Measure(q[0]):
                out_meas_z = -1.0 + out_meas_z
            else:
                out_meas_z = 1.0 + out_meas_z

        @qjit
        def testX1(q : qreg, out_meas_z: FLOAT_REF):
            X(q[1])
            if Measure(q[1]):
                out_meas_z = -1.0 + out_meas_z
            else:
                out_meas_z = 1.0 + out_meas_z

        @qjit
        def test_ref(q : qreg, out_meas_z: FLOAT_REF):
            testX0(q, out_meas_z)
            testX1(q, out_meas_z)
        
        # Note: we only support pass-by-ref for Python
        # variables in *global* scope atm.
        # i.e. no class obj member vars, etc.
        global float_result
        float_result = 0.0
        q = qalloc(2)
        test_ref(q, float_result)
        # testX0 and testX1 add -1.0 to the var.
        self.assertAlmostEqual(float_result, -2.0)

    def test_pass_by_value(self):
        @qjit
        def testX0_byVal(q : qreg, out_meas_z: float):
            X(q[0])
            if Measure(q[0]):
                out_meas_z = -1.0 + out_meas_z
            else:
                out_meas_z = 1.0 + out_meas_z

        @qjit
        def testX1_byVal(q : qreg, out_meas_z: float):
            X(q[1])
            if Measure(q[1]):
                out_meas_z = -1.0 + out_meas_z
            else:
                out_meas_z = 1.0 + out_meas_z

        @qjit
        def test_byVal(q : qreg, out_meas_z: float):
            testX0_byVal(q, out_meas_z)
            testX0_byVal(q, out_meas_z)
        
        global float_result
        float_result = 0.0
        q = qalloc(2)
        test_byVal(q, float_result)
        # No change because we pass by value
        self.assertAlmostEqual(float_result, 0.0)

    def test_bit_flip_code(self):
        @qjit
        def encodeLogicalQubit(q : qreg):
            CX(q[0], q[1])
            CX(q[0], q[2])

        @qjit
        def measureSyndrome(q : qreg, syndrome: INT_REF):
            # Make sure to clear syndrome
            syndrome = 0
            ancIdx = 3
            CX(q[0], q[ancIdx])
            CX(q[1], q[ancIdx])
            parity01 = Measure(q[ancIdx])
            if parity01: 
                # Reset anc qubit
                X(q[ancIdx])
                syndrome = syndrome + 1
            
            CX(q[1], q[ancIdx])
            CX(q[2], q[ancIdx])
            parity12 = Measure(q[ancIdx])
            if parity12:
                #Reset anc qubit
                X(q[ancIdx])
                syndrome = syndrome + 2

        @qjit
        def reset_all_qubits(q : qreg):
            for i in range(q.size()):
                if Measure(q[i]):
                    X(q[i])
        
        @qjit
        def testBitflipCode(q : qreg, qIdx: int, syndrome: INT_REF):
            H(q[0])
            encodeLogicalQubit(q)      
            # Apply error:
            if qIdx >= 0:
                X(q[qIdx])
            measureSyndrome(q, syndrome)
            reset_all_qubits(q)

        # Allocate 4 qubits: 3 qubits + 1 ancilla
        q = qalloc(4)
        global int_result
        # Init a minus value to make sure it got updated
        int_result = -1
        # No error: 
        testBitflipCode(q, -1, int_result)
        self.assertEqual(int_result, 0)
        testBitflipCode(q, 0, int_result)
        # X @ q0 -> Syndrome = 10
        self.assertEqual(int_result, 1)
        testBitflipCode(q, 1, int_result)
        # X @ q1 -> Syndrome = 11 == 3
        self.assertEqual(int_result, 3)
        testBitflipCode(q, 2, int_result)
        # X @ q2 -> Syndrome = 01 == 2
        self.assertEqual(int_result, 2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-qrt', default='ftqc')
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    unittest.main()