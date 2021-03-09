OPENQASM 3;
include "qelib1.inc";
qubit q;
bit c;

const shots = 1024;
int[32] count = 0;
int[32] count2 = 0;

for i in [0:shots] {
  h q;
  c = measure q;
  if (c == 1) {
   x q;
   count += 1;
  } else {
   count2 += 1;
  }
}

print("total count was ", count);
print("total count2 was ", count2);
