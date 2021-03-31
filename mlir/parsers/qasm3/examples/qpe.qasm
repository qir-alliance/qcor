OPENQASM 3;

const n_counting = 3;

// For this example, the oracle is the T gate 
// on the provided qubit
gate oracle b {
    t b;
}

// Inverse QFT subroutine on n_counting qubits
def iqft qubit[n_counting]:qq {
    for i in [0:n_counting/2] {
        swap qq[i], qq[n_counting-i-1];
    }
    for i in [0:n_counting-1] {
        h qq[i];
        int j = i + 1;
        int y = i;
        while (y >= 0) {
            double theta = -pi / (2^(j-y));
            cphase(theta) qq[j], qq[y];
            y -= 1;
        }
    }
    h qq[n_counting-1];
}

// Define some counting qubits
qubit counting[n_counting];

// Allocate the qubit we'll 
// put the initial state on
qubit state;

// We want T |1> = exp(2*i*pi*phase) |1> = exp(i*pi/4)
// compute phase, should be 1 / 8;

// Initialize to |1>
x state;

// Put all others in a uniform superposition
h counting;

// Loop over and create ctrl-U**2k
int repetitions = 1;
for i in [0:n_counting] {
    ctrl @ pow(repetitions) @ oracle counting[i], state;
    repetitions *= 2;
}

// Run inverse QFT 
iqf2 counting;

// Now lets measure the counting qubits
bit c[n_counting];
measure counting -> c;

// Backend is QPP which is lsb, 
// so return should be 100
print(c);
