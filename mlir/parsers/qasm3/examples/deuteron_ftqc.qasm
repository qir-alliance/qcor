
OPENQASM 3;
include "stdgates.inc";

const shots = 1024;

def deuteron(float[64]:theta) qubit[2]:q -> float[64] {
    bit first, second;
    float[64] num_parity_ones = 0.0;
    float[64] result;
    for i in [0:shots] {
        x q[0];
        ry(theta) q[1];
        cx q[1], q[0];

        h q;

        first = measure q[0];
        second = measure q[1];

        if (first != second) {
            num_parity_ones += 1.0;
        }

        reset q;
    }

    // Compute expectation value
    result = (shots - num_parity_ones) / shots - num_parity_ones / shots;
    return result;
}

float[64] theta, result, avg;
qubit qq[2];

int[32] n_trials = 10;
for i in [0:n_trials] {
  result = deuteron(theta) qq;
  avg += result;
  print("<X0X1> = ", result, avg);
}

avg /= n_trials;
print("Avg <X0X1> = ", avg);