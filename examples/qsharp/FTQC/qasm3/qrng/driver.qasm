OPENQASM 3;

// Declare kernel:
// This one is from Q#...
kernel QCOR__GenerateRandomInt__Interop(int[64]) -> int64_t;


// Generate the random number (4 bits)
int64_t max_bits = 4;
int64_t n = QCOR__GenerateRandomInt__Interop(max_bits);

// Print the random number
print("[OpenQASM3]Random int (lsb): ", n);