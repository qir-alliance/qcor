import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.xacc')

from qcor import *
# Get the NLOpt Optimzier
observable = X(0)*X(1) + Z(0) 
print("Hamiltonian = ", observable.toString())
optimizer = createOptimizer('nlopt')
