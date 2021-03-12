namespace XACC 
{
open Microsoft.Quantum.Intrinsic;
operation Ansatz(qubits : Qubit[], angles : Double[]) : Unit {
    for qubit in qubits {
        H(qubit);
    }

    for index in 0 .. Length(qubits) - 1 {
        Rx(angles[index], qubits[index]);
    }
    
    mutable results = new (Int, Result)[Length(qubits)];
    for index in 0 .. Length(qubits) - 1 {
        set results += [(index-1, M(qubits[index]))];
    }
}
}