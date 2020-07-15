
// Define a multi-register kernel
__qpu__ void bell_multi(qreg q, qreg r) {
  using qcor::xasm;
  H(q[0]);
  CX(q[0], q[1]);

  using qcor::openqasm;

  h r[0];
  cx r[0], r[1];

  using qcor::xasm;
  
  for (int i = 0; i < q.size(); i++) {
    Measure(q[i]);
    Measure(r[i]);
  }
}

int main() {

  // Create two qubit registers, each size 2
  auto q = qalloc(2);
  auto r = qalloc(2);

  // Run the quantum kernel
  bell_multi(q, r);

  // dump the results
  q.print();
  r.print();
}