#include <qcor_hadamard_test>

__qpu__ void x_gate(qreg q) { X(q[0]); }
__qpu__ void h_gate(qreg q) { H(q[0]); }

int main() {
  int n_state_qubits = 1;
  auto expectation = qcor::hadamard_test(x_gate, x_gate, n_state_qubits);
  print("<X> = ", expectation);
  
  expectation = qcor::hadamard_test(x_gate, h_gate, n_state_qubits);
  print("<H> = ", expectation);
  return 0;
}