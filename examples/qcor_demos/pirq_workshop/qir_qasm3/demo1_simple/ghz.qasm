// Compile and run with 
// qcor ghz.qasm -o ghz.x
// ./ghz.x -qrt nisq -shots 1000

OPENQASM 3;

const n_qubits = 8;

qubit q[n_qubits];

h q[0];
for i in [0:n_qubits-1] {
    ctrl @ x q[i], q[i+1];
}

bit c[n_qubits];
c = measure q;