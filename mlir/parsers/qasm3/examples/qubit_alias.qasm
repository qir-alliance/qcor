OPENQASM 3;
include "qelib1.inc";
qubit q[6];
// myreg[0] refers to the qubit q[1], myreg[1] -> q[3], etc.
let myreg = q[1, 3, 5];

for i in [0:3] {
  x q[i];
}