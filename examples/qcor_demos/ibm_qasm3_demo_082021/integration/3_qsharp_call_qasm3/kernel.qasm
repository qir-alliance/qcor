#pragma no_entrypoint;

OPENQASM 3;

#pragma export; 
def qasm_x qubit:qb {
  print("Hello from QASM3!");
  x qb;
}
