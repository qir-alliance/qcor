#pragma no_entrypoint;

OPENQASM 3;
include "stdgates.inc";

def qasm_x qubit:qb {
  x qb;
}

def qasm_h qubit:qb {
  h qb;
}