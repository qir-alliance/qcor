from qcor import * 

# Solving deuteron problem using qsim QAOA

# Define the deuteron hamiltonian 
H = -2.1433 * X(0) * X(1) - 2.1433 * \
    Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1) + 5.907

problemModel = QuaSiMo.ModelFactory.createModel(H)
      
# Create the NLOpt derivative free optimizer
optimizer = createOptimizer('nlopt')

# Create the QAOA workflow with p = 8 steps
workflow = QuaSiMo.getWorkflow('qaoa', {'optimizer': optimizer, 'steps': 8})

# Execute and print the result
result = workflow.execute(problemModel)
energy = result['energy']
print(energy)