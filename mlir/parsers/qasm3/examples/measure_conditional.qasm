OPENQASM 3;
include "qelib1.inc";

qubit q[2];
const bits_precision = 4;
bit c[bits_precision];

// Prepare the eigen-state: |1>
x q[1];

// First bit
h q[0];
// Controlled rotation: CU^k
for i in [0:8] {
  cphase(-5*pi/8) q[0], q[1];
}
h q[0];
// Measure and reset
measure q[0] -> c[0];
reset q[0];

// Second bit
h q[0];
for i in [0:4] {
  cphase(-5*pi/8) q[0], q[1];
}
// Conditional rotation
if (c[0] == 1) {
  rz(-pi/2) q[0];
}
h q[0];
// Measure and reset
measure q[0] -> c[1];
reset q[0];

// Third bit
h q[0];
for i in [0:2] {
  cphase(-5*pi/8) q[0], q[1];
}
// Conditional rotation
if (c[0] == 1) {
  rz(-pi/4) q[0];
}
if (c[1] == 1) {
  rz(-pi/2) q[0];
}
h q[0];
// Measure and reset
measure q[0] -> c[2];
reset q[0];

// Fourth bit
h q[0];
cphase(-5*pi/8) q[0], q[1];
// Conditional rotation
if (c[0] == 1) {
  rz(-pi/8) q[0];
}
if (c[1] == 1) {
  rz(-pi/4) q[0];
}
if (c[2] == 1) {
  rz(-pi/2) q[0];
}
h q[0];
measure q[0] -> c[3];