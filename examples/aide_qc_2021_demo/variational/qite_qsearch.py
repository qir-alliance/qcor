# Goals here are to just show the API, that is similar to C++
# and to show off 3rd party integration.

from qcor import *
from openfermion.ops import QubitOperator as QOp

@qjit
def state_prep(q : qreg):
    X(q[0])

qsearch_optimizer = createTransformation("qsearch")

observable = QOp('', 5.907) + QOp('Y0 Y1', -2.1433) + \
                QOp('X0 X1', -2.1433) + QOp('Z0', .21829) + QOp('Z1', -6.125) 
n_steps = 5
step_size = 0.1

problemModel = QuaSiMo.ModelFactory.createModel(state_prep, observable, 2)
workflow = QuaSiMo.getWorkflow("qite", {"steps": n_steps,
                           "step-size":step_size, "circuit-optimizer": qsearch_optimizer})

set_verbose(True)
result = workflow.execute(problemModel)

energy = result["energy"]
finalCircuit = result["circuit"]
print("\n{}\nEnergy={}\n".format(finalCircuit.toString(), energy))