#pragma no_entrypoint;

OPENQASM 3;
include "stdgates.inc";

#pragma { export; }
def qasm_x qubit:qb {
  print("Hello from QASM3!");
  x qb;
}
