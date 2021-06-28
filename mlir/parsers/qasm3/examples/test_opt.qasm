OPENQASM 3;
include "qelib1.inc";

const n = 2;

qubit q[n];

x q[0];
ry(1.2345) q[1];
y q[1];
x q[1];