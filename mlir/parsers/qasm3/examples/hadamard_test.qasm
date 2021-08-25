OPENQASM 3;
include "stdgates.inc";
const n_iters = 100;
int count1 = 0;
qubit q, a;
 
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

print("one count = ", count1);