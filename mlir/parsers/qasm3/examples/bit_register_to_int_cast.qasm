OPENQASM 3;
include "qelib1.inc";
bit c[4] = "1111";
int[4] t = int[4](c);
// should print 15
print(t);