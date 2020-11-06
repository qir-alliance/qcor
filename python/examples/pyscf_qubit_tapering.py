from qcor import *
H = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': 'H  0.000000   0.0      0.0\nH   0.0        0.0  .7474'})
print('\nOriginal:\n', H.toString())
H_tapered = operatorTransform('qubit-tapering', H)
print('\nTapered:\n', H_tapered)
