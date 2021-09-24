#include "qcor.hpp"
#include "xacc_service.hpp"

// qcor qrt_demo.cpp
// ./a.out
// qcor qrt_demo.cpp -qpu aer:ibmq_jakarta
// ./a.out

using namespace quantum;

int main() {
  // Hard code the shots for NISQ to 100
  qcor::set_shots(100);

  {  // NISQ Mode Execution

    // Allocate some qubits
    auto q = qalloc(2);

    // Get local reference to global qrt_impl pointer
    QuantumRuntime& local_qrt = *qrt_impl.get();

    // Call the quantum gate instructions on the runtime
    // for NISQ mode, this will build up a CompositeInstruction
    local_qrt.h(q[0]);
    local_qrt.cnot(q[0], q[1]);
    for (auto qi : q) local_qrt.mz(qi);

    // Submit the CompositeInstruction (just like with xacc)
    local_qrt.submit(q.results());

    // Print the results
    for (auto [bit, count] : q.counts()) {
      print(bit, ":", count);
    }
  }

  // Now switch to the FTQC runtime mode
  // must initialize with a name for the CompositeInstruction
  qrt_impl = xacc::getService<QuantumRuntime>("ftqc");
  qrt_impl->initialize("ftqc_test");

  {  // FTQC Mode Execution

    // Allocate the qubits
    auto q = qalloc(2);

    // Get local reference to the global qrt_impl pointer
    // For FTQC, we have to provide the AcceleratorBuffer
    QuantumRuntime& local_qrt = *qrt_impl.get();
    local_qrt.set_current_buffer(q.results());

    // Loop over sequential application of the
    // gate instructions, get measurement feedback, and
    // reset the state of the qubits.
    for (int i = 0; i < 20; i++) {
      local_qrt.h(q.head());
      local_qrt.cnot(q.head(), q.tail());
      auto r0 = local_qrt.mz(q[0]);
      auto r1 = local_qrt.mz(q[1]);

      if (r0 == r1) {
        print(i, "matched!");
      } else {
        print(i, "error!");
      }

      local_qrt.reset(q[0]);
      local_qrt.reset(q[1]);
    }
  }

  {  // Convenience API for these calls

    auto q = qalloc(2);
    ::quantum::h(q[0]);
    ::quantum::cnot(q[0], q[1]);
  }

  // See how we use this with the Clang SH
  // Uncomment and paste to terminal
//   printf "__qpu__ void f(qreg q) {
//       H(q[0]);
//   Measure(q[0]);
// }
// int main() {
//   auto q = qalloc(1);
//   f(q);
//   q.print();
// }
// " | qcor -print-csp-source -x c++ -

}