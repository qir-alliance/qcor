#include "qcor.hpp"
#include "xacc.hpp"

// Demo steps (show off backend extensibility via XACC)
//
// qcor xacc_demo.cpp
// ./a.out 
// ./a.out --execute
// ./a.out --accelerator aer
// ./a.out --accelerator ibm:ibmq_jakarta
// ./a.out --accelerator aer:ibmq_jakarta --execute
// ./a.out --accelerator honeywell:HQS-LT-S1-APIVAL

using namespace qcor::arg;

int main(int argc, char** argv) {
  // Command line argument setup
  std::string default_qpu = "qpp";
  add_argument("--accelerator").default_value(default_qpu);
  add_argument("--execute").default_value(false).implicit_value(true);
  add_argument("--shots").default_value(100).action(
      [](const std::string& value) { return std::stoi(value); });
  parse_args(argc, argv);

  // Get user input
  auto qpu = get_argument("--accelerator");
  auto shots = get_argument<int>("--shots");
  auto exec = get_argument<bool>("--execute");
  print("You chose the", qpu, "accelerator.");

  // This is the key piece. Ultimately under the hood 
  // for OpenQASM3, we delegate to a call like this 
  auto backend = xacc::getAccelerator(qpu, {{"shots", shots}});

  // EXAMPLES, Bell State and GHZ State
  // Execute or Print Native Code

  {  // BELL STATE
    const int n_qubits = 2;
    xacc::IRBuilder builder;
    builder.h(0);
    builder.cnot(0, 1);
    for (int i = 0; i < n_qubits; i++) builder.mz(i);
    auto composite = builder.to_ir();

    // Exec if told to, else print the native code
    if (exec) {
      auto accelerator_buffer =
          std::make_shared<xacc::AcceleratorBuffer>("q", n_qubits);
      backend->execute(accelerator_buffer, composite);
      accelerator_buffer->print();
    } else {
      print("Native Code:\n", backend->getNativeCode(composite, {{"format", "qasm"}}));
    }
  }

  {  // GHZ STATE
    const int n_qubits = 3;
    xacc::IRBuilder builder;
    builder.h(0);
    for (int i = 0; i < n_qubits - 1; i++) builder.cnot(i, i + 1);
    for (int i = 0; i < n_qubits; i++) builder.mz(i);
    auto composite = builder.to_ir();

    // Exec if told to, else print the native code
    if (exec) {
      auto accelerator_buffer =
          std::make_shared<xacc::AcceleratorBuffer>("q", n_qubits);
      backend->execute(accelerator_buffer, composite);
      accelerator_buffer->print();
    } else {
      print("Native Code:\n", backend->getNativeCode(composite, {{"format", "qasm"}}));
    }
  }

  return 0;
}
