namespace QCOR 
{
open Microsoft.Quantum.Intrinsic;

operation Bell(qubits : Qubit[]) : Unit {
    H(qubits[0]);
    for index in 0 .. Length(qubits) - 2 {
        CNOT(qubits[index], qubits[index + 1]);
    }

    for qubit in qubits {
        let res = M(qubit);
    }
}
}