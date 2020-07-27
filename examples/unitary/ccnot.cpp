
// Define a quantum kernel that uses 
// unitary matrix definition for jit-compile 
// time decomposition.
__qpu__ void ccnot(qreg q) {

  // set initial state to 111
  for (int i = 0; i < q.size(); i++) {
      X(q[i]);
  }

  // To program at the unitary matrix level, 
  // simply indicate you are using qcor::unitary namespace
  using qcor::unitary;
  
  // Create the unitary matrix
  qcor::UnitaryMatrix ccnot = qcor::UnitaryMatrix::Identity(8, 8);
  ccnot(6, 6) = 0.0;
  ccnot(7, 7) = 0.0;
  ccnot(6, 7) = 1.0;
  ccnot(7, 6) = 1.0;

  // Switch back to xasm to add some measures
  using qcor::xasm;
  for (int i = 0; i < q.size(); i++) {
      Measure(q[i]);
  }
}

int main() {
    // allocate 3 qubits
    auto q = qalloc(3);

    // By default this uses qfast with adam optimizer 
    // print what the unitary decomp was
    ccnot::print_kernel(std::cout, q);

    // Run the unitary evolution.
    ccnot(q);

    // should see 011 (msb) for toffoli input 111
    q.print();
}