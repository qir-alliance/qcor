from qcor import *
import math

# python3 test_ftqc.py -qrt ftqc

# Note: Must use FTQC runtime to get out_meas_z
@qjit
def test(q : qreg, out_meas_z: FLOAT_REF):
    H(q[0])
    if Measure(q[0]):
        out_meas_z = -1.0
    else:
        out_meas_z = 1.0

q = qalloc(1)
result = 0.0
test(q, result)
# Flipping 1.0; -1.0 (50-50)
print("Result =", result)
