from qcor import *

exp_args = [adag(0) * a(1) - adag(1)*a(0), adag(0)*a(2) - adag(2)*a(0)]

builder = KernelBuilder()

builder.x(0)
for i, exp_arg in enumerate(exp_args):
    builder.exp(('x',i), exp_arg)
ansatz = builder.create()

# Custom arg_translator in a Pythonic way
def ansatz_translate(self, q: qreg, x: List[float]):
    ret_dict = {}    
    ret_dict["q"] = q
    ret_dict["x"] = x
    ret_dict["exp_args"] = exp_args
    return ret_dict

ansatz.translate = MethodType(ansatz_translate, qjit)

# Define the hamiltonian
H = createOperator(
    '5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + 9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2')

# Create the ObjectiveFunction, specify central finite diff gradient
obj = createObjectiveFunction(
    ansatz, H, 2, {'gradient-strategy': 'central', 'step': 1e-2})

# create the lbfgs optimizer
optimizer = createOptimizer('nlopt', {'algorithm': 'l-bfgs', 'ftol': 1e-3})

# Run VQE...
results = optimizer.optimize(obj)
print(results)

#print(ansatz.llvm_mlir(qalloc(3), results[1]))