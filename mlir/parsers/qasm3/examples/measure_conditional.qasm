OPENQASM 3;
include "qelib1.inc";

qubit q;
bit c;
h q;
c = measure q;
if (c) {
    x q;
}
