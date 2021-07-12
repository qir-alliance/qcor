// qcor bell.qasm -o bell.x
// ./bell.x -qrt ftqc

OPENQASM 3;

qubit q[2];

const shots = 100;
int count = 0;

for i in [0:shots] {
    
    // Create Bell state
    h q[0];
    ctrl @ x q[0], q[1];
    
    // Measure and assert both are equal
    bit c[2];
    c = measure q;
    if (c[0] == c[1]) {
        count += 1;
    }

    reset q;
}

print("count is ", count);

