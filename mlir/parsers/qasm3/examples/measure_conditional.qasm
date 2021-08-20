OPENQASM 3;
include "qelib1.inc";

qubit q[2];
bit c[4];

h q[0];
x q[1];

// Prepare the state:
for i in [0:8] {
    cphase(-5*pi/8) q[0], q[1];
}
h q[0];

// Measure and reset
measure q[0] -> c[0];
reset q[0];

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