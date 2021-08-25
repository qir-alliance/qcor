OPENQASM 3;

// QCOR can scan for this and make 
// sure MLIRGen does not add main()
// This is useful for QASM3 files that are 
// purely library functions
#pragma no_entrypoint;

// Inverse QFT subroutine on n_counting qubits
def inverse_qft(int[64]:nc) qubit[nc]:qq {
    for i in [0:nc/2] {
        swap qq[i], qq[nc-i-1];
    }
    for i in [0:nc-1] {
        h qq[i];
        int j = i + 1;
        int y = i;
        while (y >= 0) {
            double theta = -pi / (2^(j-y));
            cphase(theta) qq[j], qq[y];
            y -= 1;
        }
    }
    h qq[nc-1];
}