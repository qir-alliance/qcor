OPENQASM 3;
include "qelib1.inc";

const n = 2;

qubit q[n];

// Rz can be moved forward (pass CX) to combine with z
rz(1.2345) q[0];
cx q[0], q[1];
z q[0];
