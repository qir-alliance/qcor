// qcor -qdk-version 0.17.2106148041-alpha qrng.qs driver.qasm
// for i in {1..10} ; do ./a.out ; done 
OPENQASM 3;

// Declare kernel:
// This one is from Q#...
kernel QCOR__GenerateRandomInt__body(int[64]) -> int64_t;


// Generate the random number (4 bits)
int64_t max_bits = 4;
int64_t n = QCOR__GenerateRandomInt__body(max_bits);

// Print the random number
print("[ OpenQASM3 ] Random", max_bits, "bit int =     ", n);