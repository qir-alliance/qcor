from qiskit.aqua.operators import (X, Y, Z, I, CX, H, ListOp, CircuitOp, Zero, EvolutionFactory,
                                   EvolvedOp, PauliTrotterEvolution, QDrift, Trotter, Suzuki)
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import transpile
from statistics import mean, stdev

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
n_runs = 10

import time
def X_op(idxs, n_qubits):
  op = None
  if 0 in idxs:
    op = X
  else:
    op = I
  for i in range(1, n_qubits):
    if (i in idxs):
      op ^= X
    else:
      op ^= I
  return op

def Z_op(idxs, n_qubits):
  op = None
  if 0 in idxs:
    op = Z
  else:
    op = I
  for i in range(1, n_qubits):
    if (i in idxs):
      op ^= Z
    else:
      op ^= I
  return op

def heisenberg_ham(n_qubits):
  Jz = 1.0
  h = 1.0
  H = -h * X_op([0], n_qubits)
  for i in range(1, n_qubits):
    H = H - h * X_op([i], n_qubits)
  for i in range(n_qubits - 1):
    H = H - Jz * (Z_op([i, i + 1], n_qubits))
  return H

n_qubits = [10, 20, 50, 100]
nbSteps = 100

def trotter_circ(q, exp_args, n_steps):
  qc = QuantumCircuit(q)
  for i in range(n_steps):
    for sub_op in exp_args:
      qc += PauliTrotterEvolution().convert(EvolvedOp(sub_op)).to_circuit()
  return qc

n_qubits = [5, 10, 20, 30, 40, 50]
nbSteps = 100

for nbQubits in n_qubits:
  data = []
  for run_id in range(n_runs):  
    ham_op = heisenberg_ham(nbQubits)
    q = QuantumRegister(nbQubits, 'q')
    start = time.time()
    comp = trotter_circ(q, ham_op.oplist, nbSteps)
    comp = transpile(comp, optimization_level=3)
    end = time.time()
    data.append(end - start)
    #ops_count = comp.count_ops()
    # num_gates = 0
    # # Count gates except identity
    # for gate_name in ops_count:
    #   if gate_name != "id":
    #     num_gates += ops_count[gate_name]
    #print("n_qubits =", nbQubits, "; n instructions =", num_gates, "; Kernel eval time:", end - start, " [secs]")
  print('n_qubits =', nbQubits, '; Elapsed time =', mean(data), '+/-', stdev(data),  '[secs]')
