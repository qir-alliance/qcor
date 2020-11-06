from qcor import *
import math

# python3 test_ftqc.py -qrt ftqc

@qjit
def testH0(q : qreg, out_meas_z: FLOAT_REF):
    print("Test H0: Input: ", out_meas_z)
    H(q[0])
    if Measure(q[0]):
        out_meas_z = -1.0 + out_meas_z
    else:
        out_meas_z = 1.0 + out_meas_z
    print("Test H0: Output: ", out_meas_z)

@qjit
def testH1(q : qreg, out_meas_z: FLOAT_REF):
    print("Test H1: Input: ", out_meas_z)
    H(q[1])
    if Measure(q[1]):
        out_meas_z = -1.0 + out_meas_z
    else:
        out_meas_z = 1.0 + out_meas_z
    print("Test H1: Output: ", out_meas_z)

# Note: Must use FTQC runtime to get out_meas_z
@qjit
def test(q : qreg, out_meas_z: FLOAT_REF):
    testH0(q, out_meas_z)
    testH1(q, out_meas_z)

q = qalloc(2)
result = 0.0
test(q, result)
print("Result =", result)
