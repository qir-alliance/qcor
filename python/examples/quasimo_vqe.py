from qcor import * 

# Define the deuteron hamiltonian 
H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

# Define the quantum kernel by providing a 
# python function that is annotated with qjit for 
# quantum just in time compilation
@qjit
def ansatz(q : qreg, theta : float):
    X(q[0])
    Ry(q[1], theta)
    CX(q[1], q[0])

# Create the problem model, provide the state 
# prep circuit, Hamiltonian and note how many qubits
# and variational parameters 
num_params = 1
problemModel = QuaSiMo.ModelFactory.createModel(ansatz, H, num_params)
      
# Create the NLOpt derivative free optimizer
optimizer = createOptimizer('nlopt')

# Create the VQE workflow
workflow = QuaSiMo.getWorkflow('vqe', {'optimizer': optimizer})

# Execute and print the result
result = workflow.execute(problemModel)
energy = result['energy']
print(energy)