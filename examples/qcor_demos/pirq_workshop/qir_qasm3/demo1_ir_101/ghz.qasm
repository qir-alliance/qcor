// Compile and run with 
// qcor ghz.qasm -o ghz.x
// 
// Run on QPP (perfect simulator)
// ./ghz.x -qrt nisq -shots 1000
//
// Kick off Hardware Executions (will have to wait, kick off in parallel)
//
// Show on real hardware (IBM)
// ./ghz.x -qpu ibm:ibmq_sydney -qrt nisq -shots 1000
//
// Show on IonQ QPU
// ./ghz.x -qpu ionq:qpu -qrt nisq -shots 1000
//
// (In meantime, show of execution on simulators)
//
// Show on IonQ remote simulator
// ./ghz.x -qpu ionq -qrt nisq -shots 1000
//
// Show on IBM remote simulator
// ./ghz.x -qpu ibm -qrt nisq -shots 1000
// 
// Show on local aer with noisy backend
// ./ghz.x -qpu aer:ibmq_sydney -qrt nisq -shots 100
//
// (Now Show off how this works, MLIR + QIR)
//
// qcor -v ghz.qasm -o ghz.x
// qcor --emit-mlir ghz.qasm
// qcor --emit-llvm ghz.qasm

OPENQASM 3;

const n_qubits = 8;

qubit q[n_qubits];

h q[0];
for i in [0:n_qubits-1] {
    ctrl @ x q[i], q[i+1];
}

bit c[n_qubits];
c = measure q;