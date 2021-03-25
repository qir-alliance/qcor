OPENQASM 3;

// Global constant, maximum bit size 
// for the random integer
const max_bits = 4;

// Generate a superposition and 
// measure to return a 50/50 random bit
def random_bit() qubit:a -> bit {
    h a;
    return measure a;
}

// Generate a random integer of max_bits bit width
// This will generate a random 0 or 1 
// based on a single provided qubit put 
// in a superposition
def generate_random_int() qubit:q -> int {
    // Create [0,0,0,...0] of size max_bits
    bit b[max_bits];

    // Set every bit as a random 0 or 1
    for i in [0:max_bits] {
        b[i] = random_bit() q;
        // reset qubit state for 
        // next iteration
        reset q;
    }
    // Print the binary string
    print("random binary: ", b);
    int n = int[32](b);
    return n;
}

// Allocate a single qubit
qubit a;

// Generate the random number 
// using the allocated qubit
int n = generate_random_int() a;

// print the random number
print("Random int (lsb): ", n);