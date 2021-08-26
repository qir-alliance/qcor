OPENQASM 3;

// compute < psi | U | psi> 
// let |psi> = |+>, U = X
// < + | X | + > == 1.0
// == < 0 | H X H | 0> == < 0 | Z | 0 > == 1

// compile and run with 
// qcor hadamard_test.qasm 
// ./a.out

const n_iters = 100;
double count1;

// Ancilla qubit
qubit a;

// Eigenstate qubit
qubit q;

for i in [0:n_iters] {
    // Generate |+> eigenstate
    h q;

    // apply hadamard on ancilla
    h a;

    // Ctrl-U, U == X
    cx a, q;

    // apply hadamard again
    h a;

    // measure the ancilla
    bit c;
    c = measure a;

    // Reset the qubits
    reset a;
    reset q;

    // Store up the observed bits
    if (c == 1) {
      count1 = count1 + 1;
    } 
}

// Number of 0s must be ...
double count0 = n_iters - count1;

// compute the exp val Re(<psi|U|psi>) = P0 - P1
double exp_val = (count0 - count1) / n_iters;
print("<+|X|+> = ", exp_val);