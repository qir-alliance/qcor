
__qpu__ void ccnot(qreg q) {

  // set initial state to 111
  X(q);

  X::ctrl({q[0], q[1]}, q[2]);
  
  Measure(q);
}

int main() {
  // allocate 3 qubits
  auto q = qalloc(3);

  ccnot::print_kernel(std::cout, q);

  // Run the unitary evolution.
  ccnot(q);

  // should see 011 (msb) for toffoli input 111
  q.print();
}

