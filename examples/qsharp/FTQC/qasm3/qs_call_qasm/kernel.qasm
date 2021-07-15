OPENQASM 3;
include "stdgates.inc";

def qasm_x qubit:qb {
  print("Call X gate from QASM3!");
  x qb;
}
