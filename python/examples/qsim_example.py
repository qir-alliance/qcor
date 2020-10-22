import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + "/.xacc")

from qcor import *
import numpy as np

# Set up QCOR runtime
Initialize(qpu="qpp")

# Time-dependent Hamiltonian: 
# Returns the Pauli operators at a time point.
def td_hamiltonian(t):
  Jz = 2 * np.pi * 2.86265 * 1e-3
  epsilon = Jz
  omega = 4.8 * 2 * np.pi * 1e-3
  return -Jz * Z(0) * Z(1)  - Jz * Z(1) * Z(2) + (-epsilon * np.cos(omega * t)) * (X(0) + X(1) + X(2)) 

observable = (1.0 / 3.0) * (Z(0) + Z(1) + Z(2))
print("observable = ", observable.toString())
# Example: build model and TD workflow for Fig. 2 of
# https://journals.aps.org/prb/pdf/10.1103/PhysRevB.101.184305
problemModel = qsim.ModelBuilder.createModel(observable, td_hamiltonian)
# TD workflow with hyper-parameters: 
# Trotter step = 3fs, number of steps = 100 -> end time = 300fs
workflow = qsim.getWorkflow(
      "td-evolution", {"method": "trotter", "dt": 3.0, "steps": 100})

# Result contains the observable expectation value along Trotter steps.
result = workflow.execute(problemModel)