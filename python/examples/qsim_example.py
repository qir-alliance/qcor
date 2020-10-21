import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.xacc')

from qcor import *

# Time-dependent Hamiltonian: 
# Returns the Pauli operators at a time point.
def td_hamiltonian(t):
  omega = 1.0
  print("HOWDY: Python callback")
  return X(0) + X(1) + X(2)


# This is for testing-purposes only
observable = X(0)*X(1) + Z(0) + Z(1)
print("observable = ", observable.toString())
optimizer = createOptimizer('nlopt')
model = qsim.ModelBuilder.createModel(observable, td_hamiltonian)