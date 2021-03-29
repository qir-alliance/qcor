namespace QCOR 
{
open QCOR.Intrinsic;
// Estimate energy value in a FTQC manner.
// Passing Array type b/w C++ driver and Q#
operation Ansatz(angles : Double[], shots: Int) : Double {
    mutable numParityOnes = 0;
    use qubits = Qubit[2]
    {
        for test in 1..shots {
            Rx(angles[0], qubits[0]);
            Rx(angles[1], qubits[1]);
            CNOT(qubits[1], qubits[0]);
            // Let's measure <X0X1>
            H(qubits[0]);
            H(qubits[1]);
            if M(qubits[0]) != M(qubits[1]) 
            {
                set numParityOnes += 1;
            }
            Reset(qubits[0]);
            Reset(qubits[1]);
        }
    }
    let res =  IntAsDouble(shots - numParityOnes)/IntAsDouble(shots) - IntAsDouble(numParityOnes)/IntAsDouble(shots);
    return res;
}
}