OPENQASM 3;
include "stdgates.inc";

const shots = 1024;
// State-preparation:
def ansatz(float[64]:theta) qubit[2]:q {
    x q[0];
    ry(theta) q[1];
    cx q[1], q[0];
}

def deuteron(float[64]:theta) qubit[2]:q -> float[64] {
    bit first, second;
    float[64] num_parity_ones = 0.0;
    float[64] result;
    for i in [0:shots] {
        ansatz(theta) q;
        // Change measurement basis
        h q;
        // Measure
        first = measure q[0];
        second = measure q[1];
        if (first != second) {
            num_parity_ones += 1.0;
        }
        // Reset
        reset q;
    }

    // Compute expectation value
    result = (shots - num_parity_ones) / shots - num_parity_ones / shots;
    return result;
}

float[64] theta, exp_val;
qubit qq[2];
// Try a theta value:
theta = 0.123;
exp_val = deuteron(theta) qq;
print("Avg <X0X1> = ", exp_val);