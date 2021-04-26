
namespace QCOR {
    open QCOR.Intrinsic;
    operation SWAP(q1 : Qubit, q2: Qubit) : Unit is Adj {
        CNOT(q1, q2);
        CNOT(q2, q1);
        CNOT(q1, q2);
    }
    
    // 1-qubit QFT
    operation OneQubitQFT (q : Qubit) : Unit is Adj {
        H(q);
    }
    // Rotation gate
    // Applies a rotation about the |1⟩ state by an angle 
    // specified as a dyadic fraction.
    operation Rotation (q : Qubit, k : Int) : Unit is Adj+Ctl {
        let angle = 2.0 * PI() / IntAsDouble(1 <<< k);
        Rz(angle, q);
    }
    // Prepare binary fraction exponent in place (quantum input)
    operation BinaryFractionQuantumInPlace (register : Qubit[]) : Unit is Adj {
        OneQubitQFT(register[0]);
        for ind in 1 .. Length(register) - 1 {
            Controlled Rotation([register[ind]], (register[0], ind + 1));
        }
    }
    // Reverse the order of qubits
    operation ReverseRegister (register : Qubit[]) : Unit is Adj {
        let N = Length(register);
        for ind in 0 .. N / 2 - 1 {
            SWAP(register[ind], register[N - 1 - ind]);
        }
    }
    // Quantum Fourier transform
    // Input: A register of qubits in state |j₁j₂...⟩
    // Goal: Apply quantum Fourier transform to the input register
    operation QuantumFourierTransform (register : Qubit[]) : Unit is Adj {
        let n = Length(register);
        for i in 0 .. n - 1 {
            BinaryFractionQuantumInPlace(register[i ...]);
        }
        ReverseRegister(register);
    }
    
    operation InverseQFT (register : Qubit[]) : Unit {
        Adjoint QuantumFourierTransform(register);
    }
}

