import unittest
from qcor import *

float_result = 0.0

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-qrt', default='ftqc')
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    unittest.main()