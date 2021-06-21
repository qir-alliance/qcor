OPENQASM 3;


const n_counting = 3;

// Declare external quantum subroutine:
def QCOR__IQFT__body qubit[n_counting]:qq extern;

// For this example, the oracle is the T gate 
// on the provided qubit
gate oracle b {
    t b;
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
QCOR__IQFT__body counting;

// Now lets measure the counting qubits
bit c[n_counting];
measure counting -> c;

// Backend is QPP which is lsb, 
// so return should be 100
print(c);
