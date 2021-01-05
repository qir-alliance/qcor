from qcor import *
from openfermion.ops import FermionOperator as FOp

# Define the Hamiltonian
H=  FOp('', 0.7080240981) + FOp('1^ 2^ 1 2', -0.165606823582) + FOp('1^ 0^ 0 1', 0.120200490713) + \
    FOp('0^ 3^ 1 2', -0.0454063328691) + FOp('2^ 0^ 0 2', 0.168335986252) + \
    FOp('1^ 2^ 3 0', 0.0454063328691) + FOp('0^ 2^ 2 0', 0.168335986252) + \
    FOp('0^ 3^ 3 0', 0.165606823582) + FOp('3^ 0^ 2 1', -0.0454063328691) + \
    FOp('1^ 3^ 0 2', -0.0454063328691) + FOp('3^ 1^ 2 0', -0.0454063328691) + \
    FOp('1^ 2^ 2 1', 0.165606823582) + FOp('0^ 3^ 0 3', -0.165606823582) + \
    FOp('3^ 3', -0.479677813134) + FOp('1^ 2^ 0 3', -0.0454063328691) + \
    FOp('1^ 3^ 1 3', -0.174072892497) + FOp('0^ 2^ 1 3', -0.0454063328691) + \
    FOp('0^ 1^ 1 0', 0.120200490713) + FOp('0^ 2^ 3 1', 0.0454063328691) + \
    FOp('1^ 3^ 3 1', 0.174072892497) + FOp('2^ 1^ 1 2', 0.165606823582) + \
    FOp('2^ 1^ 3 0', -0.0454063328691) + FOp('2^ 3^ 2 3', -0.120200490713) + \
    FOp('2^ 3^ 3 2', 0.120200490713) + FOp('0^ 2^ 0 2', -0.168335986252) + \
    FOp('3^ 2^ 2 3', 0.120200490713) + FOp('3^ 2^ 3 2', -0.120200490713) + \
    FOp('1^ 3^ 2 0', 0.0454063328691) + FOp('0^ 0', -1.2488468038) + \
    FOp('3^ 1^ 0 2', 0.0454063328691) + FOp('2^ 0^ 2 0', -0.168335986252) + \
    FOp('3^ 0^ 0 3', 0.165606823582) + FOp('2^ 0^ 3 1', -0.0454063328691) + \
    FOp('2^ 0^ 1 3', 0.0454063328691) + FOp('2^ 2', -1.2488468038) + \
    FOp('2^ 1^ 0 3', 0.0454063328691) + FOp('3^ 1^ 1 3', 0.174072892497) + \
    FOp('1^ 1', -0.479677813134) + FOp('3^ 1^ 3 1', -0.174072892497) + \
    FOp('3^ 0^ 1 2', 0.0454063328691) + FOp('3^ 0^ 3 0', -0.165606823582) + \
    FOp('0^ 3^ 2 1', 0.0454063328691) + FOp('2^ 1^ 2 1', -0.165606823582) + \
    FOp('0^ 1^ 0 1', -0.120200490713) + FOp('1^ 0^ 1 0', -0.120200490713) 

# Create a QSim problem model for the OpenFermion Hamiltonian
problemModel = qsim.ModelFactory.createModel(H)
      
# Create the Adapt-VQE workflow
nElectrons = 2
pool_vqe = 'qubit-pool'
optimizer = createOptimizer('nlopt', {'nlopt-optimizer': 'l-bfgs'})
workflow = qsim.getWorkflow('adapt', {'optimizer': optimizer,
                                      'pool': pool_vqe,
                                      'n-electrons': nElectrons})
# Execute and print the result
result = workflow.execute(problemModel)
energy = result['energy']
# Expected: -1.137
print('Energy = ', energy)
# Print the final adapt ansatz circuit
adapt_ansatz = result['circuit']
print('Final circuit:\n', adapt_ansatz)