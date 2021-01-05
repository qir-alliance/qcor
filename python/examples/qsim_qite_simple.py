from qcor import * 

# Simple 1-qubit demonstration of the Quatum Imaginary Time Evolution/QLanczos algorithm
# Reference: https://www.nature.com/articles/s41567-019-0704-4
# Target H = 1/sqrt(2)(X + Z)
# Expected minimum value: -1.0
H = 0.7071067811865475 * X(0) + 0.7071067811865475 * Z(0)

# See Fig. 2 (e) of https://www.nature.com/articles/s41567-019-0704-4
# Horizontal axis: 0 -> 2.5
# The number of Trotter steps 
nbSteps = 25

# The Trotter step size
stepSize = 0.1

# Create the problem model
problemModel = qsim.ModelFactory.createModel(H)

# Create the QITE workflow
workflow = qsim.getWorkflow('qite', {'steps': nbSteps, 'step-size': stepSize})

# Execute and print the result
result = workflow.execute(problemModel)

# Final energy:
energy = result['energy']
print('Ground state energy =', energy)

# Plot energy value along QITE steps
td_energy_vals = result['exp-vals']
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.arange(0, nbSteps + 1) * stepSize, td_energy_vals, 'ro-', label = 'QITE')
plt.legend()
plt.grid()
plt.savefig('qite.pdf')

