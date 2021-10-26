# Run this from the command line like this
#
# python3 exp_fermion.py -shots 100

from qcor import *
from openfermion.ops import FermionOperator as FOp

@qjit
def ansatz(q: qreg, x: List[float], exp_args: List[Operator]):
    X(q[0])
    for i, exp_arg in enumerate(exp_args):
        exp_i_theta(q, x[i], exp_args[i])

# Create OpenFermion operators for our quantum kernel...
exp_args_openfermion = [FOp('0^ 1') - FOp('1^ 0'), FOp('0^ 2') - FOp('2^ 0')]

# We need to translate OpenFermion ops into qcor Operators to use with kernels...
exp_args_qcor = [createOperator('fermion', fop) for fop in exp_args_openfermion]

ansatz.print_kernel(qalloc(3), [1.0], exp_args_qcor)