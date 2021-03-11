OPENQASM 3;
include "qelib1.inc";
qubit q;
bit c;

const shots = 1024;
int[32] ones = 0;
int[32] zeros = 0;

for i in [0:shots] {
  h q;
  c = measure q;
  if (c == 1) {
   ones += 1;
  } else {
   zeros += 1;
  }
  reset q;
}

print("N |1> measured = ", ones);
print("N |0> measured = ", zeros);
