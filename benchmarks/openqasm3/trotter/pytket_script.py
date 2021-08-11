import time
from pytket.circuit import Circuit, PauliExpBox
from pytket.pauli import Pauli
from pytket.extensions.qiskit import AerStateBackend
from pytket.passes import FullPeepholeOptimise
from statistics import mean, stdev
nb_steps = 100
step_size = 0.01
n_qubits = [5, 10, 20, 30, 40, 50]
n_runs = 10
for nb_qubits in n_qubits:
  data = []
  for run_id in range(n_runs):  
    # Start timer
    start = time.time()
    circ = Circuit(nb_qubits)

    h = 1.0
    Jz = 1.0
    for i in range(nb_steps):
        # Using Heisenberg Hamiltonian:
        for q in range(nb_qubits):
            circ.add_pauliexpbox(PauliExpBox([Pauli.X], -h * step_size), [q])
        for q in range(nb_qubits - 1):
            circ.add_pauliexpbox(PauliExpBox([Pauli.Z, Pauli.Z], -Jz * step_size), [q, q + 1])

    # Compile to gates
    backend = AerStateBackend()
    circ = backend.get_compiled_circuit(circ)

    # Apply optimization
    FullPeepholeOptimise().apply(circ)

    end = time.time()
   
    data.append(end - start)
  
  print('n_qubits =', nb_qubits, '; Elapsed time =', mean(data), '+/-', stdev(data),  '[secs]')
    # for com in circ: # equivalently, circ.get_commands()
    #   print(com.op, com.op.type, com.args)