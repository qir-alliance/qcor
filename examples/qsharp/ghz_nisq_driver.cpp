#include <iostream> 
#include <vector>
#include "qcor.hpp"
using Qubit = uint64_t;
using QReg = std::vector<Qubit*>;
// Include the external QSharp function.
qcor_include_qsharp(XACC__TestGhz__body, void, Array *);

class TestGhz : public qcor::QuantumKernel<class TestGhz, qreg> {
private:
  // Use deque to prevent re-alloc
  std::deque<Qubit> m_qubits;
  friend class qcor::QuantumKernel<class TestGhz, qreg>;

protected:
  void operator()(qreg q) {
    if (!parent_kernel) {
      parent_kernel = qcor::__internal__::create_composite(kernel_name);
    }
    quantum::set_current_program(parent_kernel);
    if (runtime_env == QrtType::FTQC) {
      quantum::set_current_buffer(q.results());
    }
    QReg qReg;
    for (int i = 0; i < q.size(); ++i) {
      m_qubits.push_back(i);
      qReg.emplace_back(&m_qubits.back());
    }
    XACC__TestGhz__body(&qReg);
    std::cout << "INVOKE:\n" << parent_kernel->toString();
  }

public:
  inline static const std::string kernel_name = "TestGhz";
  TestGhz(qreg q) : QuantumKernel<TestGhz, qreg>(q) {}
  TestGhz(std::shared_ptr<qcor::CompositeInstruction> _parent, qreg q)
      : QuantumKernel<TestGhz, qreg>(_parent, q) {}
  virtual ~TestGhz() {
    if (disable_destructor) {
      return;
    }
    auto [q] = args_tuple;
    operator()(q);
    xacc::internal_compiler::execute_pass_manager();
    if (optimize_only) {
      return;
    }
    if (is_callable) {
      quantum::submit(q.results());
    }
  }
};

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qpu aer:ibmqx2 -shots 1024 ghz_nisq.qs ghz_nisq_driver.cpp
// Run with:
// $ ./a.out
int main() {
  auto q = qalloc(3);
  qcor::set_verbose(true);
  { TestGhz kernel(q); }
  q.print();
  return 0;
}