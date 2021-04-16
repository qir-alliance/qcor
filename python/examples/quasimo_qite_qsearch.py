from qcor import *

@qjit
def state_prep(q : qreg):
    X(q[0])

qsearch_optimizer = createTransformation("qsearch")

observable = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + \
                    .21829 * Z(0) - 6.125 * Z(1) + 5.907


n_steps = 5
step_size = 0.1

problemModel = QuaSiMo.ModelFactory.createModel(state_prep, observable, 2, 0)
workflow = QuaSiMo.getWorkflow("qite", {"steps": n_steps,
                           "step-size":step_size, "circuit-optimizer": qsearch_optimizer})

set_verbose(True)
result = workflow.execute(problemModel)

energy = result["energy"]
finalCircuit = result["circuit"]
print("\n{}\nEnergy={}\n".format(finalCircuit.toString(), energy))