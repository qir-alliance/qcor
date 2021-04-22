# Run this from the command line like this
#
# python3 exp_fermion.py -shots 100

from qcor import *
from openfermion.ops import FermionOperator as FOp

@qjit
def ansatz(q: qreg, x: List[float], exp_args: List[FermionOperator]):
    X(q[0])
    for i, exp_arg in enumerate(exp_args):
        exp_i_theta(q, x[i], exp_args[i])

# Create OpenFermion operators for our quantum kernel...
exp_args_openfermion = [FOp('0^ 1') - FOp('1^ 0'), FOp('0^ 2') - FOp('2^ 0')]

# We need to translate OpenFermion ops into qcor Operators to use with kernels...
exp_args_qcor = [createOperator('fermion', fop) for fop in exp_args_openfermion]

# Define the hamiltonian
H = createOperator(
    '5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + 9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2')

def obj(x : List[float]):
    e_at_x = ansatz.observe(H, qalloc(3), x, exp_args_qcor)
    print(x,e_at_x)
    return e_at_x
    
# create the cobyla optimizer
optimizer = createOptimizer('nlopt', {'maxeval':25})

# Run VQE...
results = optimizer.optimize(obj, 2)
print(results)
