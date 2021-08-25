OPENQASM 3;

// < + | X | + > == 1.0

const n_iters = 100;
double count0, count1;

// Eigenstate qubit
qubit q;

// Ancilla qubit
qubit a;

for i in [0:n_iters] {
    // Generate |+> eigenstate
    x q;

    // apply hadamard on ancilla
    h a;

    // Ctrl-U, U == H
    cx a, q;

    // apply hadamard again
    h a;

    // measure and reset
    bit c;
    c = measure a;
    reset a;

    // Store up the observed bits
    if (c == 1) {
      count1 = count1 + 1;
    } 
}

// Number of 0s must be ...
count0 = n_iters - count1;

double exp_val = (count1 - count0) / (count1 + count0);
print("<+|H|+> = ", exp_val);