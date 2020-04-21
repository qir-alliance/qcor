#include <qalloc>

__qpu__ void teleport(qreg q) {
  // State preparation (Bob)
  X(q[0]);
  // Bell channel setup
  H(q[1]);
  CX(q[1], q[2]);
  // Alice Bell measurement
  CX(q[0], q[1]);
  H(q[0]);
  Measure(q[0]);
  Measure(q[1]);
  // Correction
  if (q[0]) {
    Z(q[2]);
  }
  if (q[1]) {
    X(q[2]);
  }
  // Measure teleported qubit
  Measure(q[2]);
}

int main() {

    // Allocate the qubits
    auto q = qalloc(3);

    // run the teleportation
    teleport(q);

    // dump the results
    q.print();

}