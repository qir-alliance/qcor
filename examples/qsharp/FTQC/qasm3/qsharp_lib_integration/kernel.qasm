#pragma no_entrypoint;

OPENQASM 3;
include "stdgates.inc";

#pragma { export; }
def qasm_x qubit:qb {
  x qb;
}

#pragma { export; }
def qasm_h qubit:qb {
  h qb;
}