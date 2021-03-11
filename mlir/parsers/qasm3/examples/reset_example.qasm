OPENQASM 3;
include "qelib1.inc";

qubit q;

const shots = 1024;
int[32] count = 0;

for i in [0:shots] {
    // put qubit in a superposition
    h q;
    
    // Reset that qubit to |0>
    reset q;

    // Set it to |1>
    x q;

    // Measure, test that it is 1
    // to ensure reset is working right
    bit c;
    c = measure q;
    if (c == 1) {
        count += 1;
    }
}

print("count is ", count);