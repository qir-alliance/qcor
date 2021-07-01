/// Simulator
/// $qcor -qpu qpp ghz.cpp 

/// IBMQ Backend (normal gate-mode)
/// $qcor -qpu ibm:ibmqx2 ghz.cpp

/// IBMQ Backend (pulse mode)
/// QCOR lowers gates --> pulses
/// $qcor -qpu ibm:ibmq_montreal[mode:pulse] ghz.cpp 

/// IonQ 
/// Simulator:
/// qcor -qpu ionq ghz.cpp
/// QPU (11 qubits)
/// qcor -qpu ionq:qpu ghz.cpp

__qpu__ void ghz(qreg q) {
  H(q[0]);

  for (int i = 0; i < q.size() - 1; i++) {
    CX(q[i], q[i + 1]);
  }

  Measure(q);
}

int main() {
  set_shots(100);
  auto q = qalloc(3);

  ghz(q);

  auto counts = q.counts();

  for (auto [bit, count] : counts) {
    print(bit, ": ", count);
  }
}