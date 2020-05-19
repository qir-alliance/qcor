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

void set_shots(int shots) {}

void one_qubit_inst(const std::string &name, const qubit &qidx,
                    std::vector<double> parameters) {
  auto inst =
      provider->createInstruction(name, std::vector<std::size_t>{qidx.second});
  inst->setBufferNames({qidx.first});
  for (int i = 0; i < parameters.size(); i++) {
    inst->setParameter(i, parameters[i]);
  }
  program->addInstruction(inst);
}

void h(const qubit &qidx) { one_qubit_inst("H", qidx); }
void x(const qubit &qidx) { one_qubit_inst("X", qidx); }

void rx(const qubit &qidx, const double theta) {
  one_qubit_inst("Rx", qidx, {theta});
}

void ry(const qubit &qidx, const double theta) {
  one_qubit_inst("Ry", qidx, {theta});
}
void rz(const qubit &qidx, const double theta) {
  one_qubit_inst("Rz", qidx, {theta});
}

void mz(const qubit &qidx) { one_qubit_inst("Measure", qidx); }

void cnot(const qubit &src_idx, const qubit &tgt_idx) {
  auto cx = provider->createInstruction(
      "CNOT", std::vector<std::size_t>{src_idx.second, tgt_idx.second});
  cx->setBufferNames({src_idx.first, tgt_idx.first});
  program->addInstruction(cx);
}

void submit(xacc::AcceleratorBuffer *buffer) {
  xacc::internal_compiler::execute(buffer, program.get());
}

void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers) {
  xacc::internal_compiler::execute(buffers, nBuffers, program.get());
}
std::shared_ptr<xacc::CompositeInstruction> getProgram() { return program; }
} // namespace quantum