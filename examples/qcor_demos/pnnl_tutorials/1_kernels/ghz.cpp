// Demonstrate the programmability of a simple kernel
// (1) Run with QPP first
// (2) Run with noisy backend (remote Honeywell simulator)
// qcor -set-credentials honeywell
// qcor ghz.cpp -qpu honeywell:HQS-LT-S1-SIM -shots 1024
// (3) Run with IonQ 
// qcor ghz.cpp -qpu ionq -shots 1024
// (4) Run with OLCF QLM simulator (QLMaaS)
// qcor ghz.cpp -qpu atos-qlm -shots 1024
// (5) Run with just print_kernel to show automatic placement
// Uncomment print_kernel
// qcor ghz.cpp -qpu ibm:ibmq_toronto

__qpu__ void ghz(qreg q) {
  H(q[0]);
  for (int i = 0; i < q.size() - 1; i++) {
    CX(q[i], q[i + 1]);
  }
  Measure(q);
}

int main() {
  auto q = qalloc(6);
  ghz(q);
  q.print();
  // ghz::print_kernel(q);
}