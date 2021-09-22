OPENQASM 3;
include "stdgates.inc";

const n_qubits = 50;
const n_layers = 12;
qubit q[n_qubits];

// Compile: qcor random_circuit.qasm
// Run by: mpiexec -n <N> ./a.out -qrt nisq -qpu tnqvm -qpu-config tnqvm.ini 

// Loop over layers
float[64] theta = 1.234;
for i in [0:n_layers] {
    // Single qubit layers:
    for j in [0:n_qubits] {
        rx(theta) q[j];
    }

    // For demonstration purposes, just change the 
    // angle in each layer by adding 1.0.
    theta += 1.0;
    // Entanglement layers:
    for j in [0:n_qubits - 1] {
        cx q[j], q[j+1];
    }
}
