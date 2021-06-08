namespace QCOR 
{
open Microsoft.Quantum.Intrinsic;
open Microsoft.Quantum.Convert;
// Estimate energy value in a FTQC manner.
@EntryPoint()
operation Deuteron(theta : Double, shots: Int) : Double {
    mutable numParityOnes = 0;
    use qubits = Qubit[2]
    {
        for test in 1..shots {
            X(qubits[0]);
            Ry(theta, qubits[1]);
            CNOT(qubits[1], qubits[0]);
            // Let's measure <X0X1>
            H(qubits[0]);
            H(qubits[1]);
            if M(qubits[0]) != M(qubits[1]) 
            {
                set numParityOnes += 1;
            }
            if M(qubits[0]) == One {
                X(qubits[0]);
            }
            if M(qubits[1]) == One {
                X(qubits[1]);
            }
        }
    }
    let res =  IntAsDouble(shots - numParityOnes)/IntAsDouble(shots) - IntAsDouble(numParityOnes)/IntAsDouble(shots);
    return res;
}
}