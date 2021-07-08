OPENQASM 3;

qubit q[2];

const shots = 1024;
int count = 0;

for i in [0:shots] {
    h q[0];
    ctrl @ x q[0], q[1];
    bit c[2];
    c = measure q;
    if (c[0] == c[1]) {
        count += 1;
    }

    reset q;
}

print("count is ", count);

