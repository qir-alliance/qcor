#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"

namespace quantum {
std::shared_ptr<xacc::CompositeInstruction> program = nullptr;
std::shared_ptr<xacc::IRProvider> provider = nullptr;

void initialize(const std::string qpu_name, const std::string kernel_name) {
  xacc::internal_compiler::compiler_InitializeXACC(qpu_name.c_str());
  provider = xacc::getIRProvider("quantum");
  program = provider->createComposite(kernel_name);
}
void set_shots( int shots ) {

}
void h(const std::size_t qidx) {
  auto hadamard =
      provider->createInstruction("H", std::vector<std::size_t>{qidx});
  program->addInstruction(hadamard);
}
void cnot(const std::size_t src_idx, const std::size_t tgt_idx) {
  auto cx = provider->createInstruction(
      "CNOT", std::vector<std::size_t>{src_idx, tgt_idx});
  program->addInstruction(cx);
}

void mz(const std::size_t qidx) {
  auto meas =
      provider->createInstruction("Measure", std::vector<std::size_t>{qidx});
  program->addInstruction(meas);
}

void submit(xacc::AcceleratorBuffer *buffer) {
  xacc::internal_compiler::execute(buffer, program.get());
}

} // namespace quantum