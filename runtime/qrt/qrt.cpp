#include "qrt.hpp"
#include "Instruction.hpp"
#include "PauliOperator.hpp"
#include "pass_manager.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"
#include <Eigen/Dense>
#include <Utils.hpp>

namespace xacc {
namespace internal_compiler {
// Extern vars:
int __opt_level = 0;
bool __print_opt_stats = false;
std::string __user_opt_passes = "";
std::string __placement_name = "";
std::vector<int> __qubit_map = {};

void execute_pass_manager() {
  qcor::internal::PassManager passManager(__opt_level, __qubit_map,
                                          __placement_name);
  auto optData = passManager.optimize(::quantum::qrt_impl->get_current_program());

  std::vector<std::string> user_passes;
  if (!__user_opt_passes.empty()) {
    std::stringstream ss(__user_opt_passes);
    // Parses list of passes
    while (ss.good()) {
      std::string passName;
      std::getline(ss, passName, ',');
      user_passes.emplace_back(passName);
    }
  }

  // Runs user-specified passes
  for (const auto &user_pass : user_passes) {
    optData.emplace_back(
        qcor::internal::PassManager::runPass(user_pass, ::quantum::qrt_impl->get_current_program()));
  }

  if (__print_opt_stats) {
    // Prints out the Optimizer Stats if requested.
    for (const auto &passData : optData) {
      std::cout << passData.toString(false);
    }
  }

  passManager.applyPlacement(::quantum::qrt_impl->get_current_program());
}

std::vector<int> parse_qubit_map(const char *qubit_map_str) {
  std::vector<int> qubitMap;
  std::stringstream ss(qubit_map_str);
  while (ss.good()) {
    // Split by ',' delimiter
    try {
      std::string qubitId;
      std::getline(ss, qubitId, ',');
      qubitMap.emplace_back(std::stoi(qubitId));
    } catch (...) {
      // Cannot parse the integer.
      return {};
    }
  }
  return qubitMap;
}

void apply_decorators(const std::string &decorator_cmdline_string) {
  auto decorator =
      xacc::getAcceleratorDecorator(decorator_cmdline_string, get_qpu());
  xacc::internal_compiler::qpu = decorator;
}
} // namespace internal_compiler
} // namespace xacc
namespace quantum {
int current_shots = 0;
std::shared_ptr<QuantumRuntime> qrt_impl = nullptr;
std::vector<std::string> kernels_in_translation_unit = {};

void initialize(const std::string qpu_name, const std::string kernel_name) {
  xacc::internal_compiler::compiler_InitializeXACC(qpu_name.c_str());

  qrt_impl = xacc::getService<QuantumRuntime>("nisq");
  qrt_impl->initialize(kernel_name);
}

void set_backend(std::string accelerator_name, const int shots) {

  xacc::internal_compiler::compiler_InitializeXACC(accelerator_name.c_str());
  set_shots(shots);
}

void set_backend(std::string accelerator_name) {
  xacc::internal_compiler::compiler_InitializeXACC(accelerator_name.c_str());
}

void set_shots(int shots) {
  current_shots = shots;
  xacc::internal_compiler::get_qpu()->updateConfiguration(
      {std::make_pair("shots", shots)});
}

int get_shots() { return current_shots; }

void h(const qubit &qidx) { qrt_impl->h(qidx); }
void x(const qubit &qidx) { qrt_impl->x(qidx); }
void y(const qubit &qidx) { qrt_impl->y(qidx); }
void z(const qubit &qidx) { qrt_impl->z(qidx); }

void s(const qubit &qidx) { qrt_impl->s(qidx); }
void sdg(const qubit &qidx) { qrt_impl->sdg(qidx); }

void t(const qubit &qidx) { qrt_impl->t(qidx); }
void tdg(const qubit &qidx) { qrt_impl->tdg(qidx); }

void rx(const qubit &qidx, const double theta) { qrt_impl->rx(qidx, theta); }

void ry(const qubit &qidx, const double theta) { qrt_impl->ry(qidx, theta); }

void rz(const qubit &qidx, const double theta) { qrt_impl->rz(qidx, theta); }

void u1(const qubit &qidx, const double theta) { qrt_impl->u1(qidx, theta); }

void u3(const qubit &qidx, const double theta, const double phi,
        const double lambda) {
  qrt_impl->u3(qidx, theta, phi, lambda);
}

void mz(const qubit &qidx) { qrt_impl->mz(qidx); }

void cnot(const qubit &src_idx, const qubit &tgt_idx) {
  qrt_impl->cnot(src_idx, tgt_idx);
}

void cy(const qubit &src_idx, const qubit &tgt_idx) {
  qrt_impl->cy(src_idx, tgt_idx);
}

void cz(const qubit &src_idx, const qubit &tgt_idx) {
  qrt_impl->cz(src_idx, tgt_idx);
}

void ch(const qubit &src_idx, const qubit &tgt_idx) {
  qrt_impl->ch(src_idx, tgt_idx);
}

void swap(const qubit &src_idx, const qubit &tgt_idx) {
  qrt_impl->swap(src_idx, tgt_idx);
}

void cphase(const qubit &src_idx, const qubit &tgt_idx, const double theta) {
  qrt_impl->cphase(src_idx, tgt_idx, theta);
}

void crz(const qubit &src_idx, const qubit &tgt_idx, const double theta) {
  qrt_impl->crz(src_idx, tgt_idx, theta);
}

void exp(qreg q, const double theta, xacc::Observable &H) {
  qrt_impl->exp(q, theta, H);
}

void exp(qreg q, const double theta, xacc::Observable *H) {
  qrt_impl->exp(q, theta, H);
}

void exp(qreg q, const double theta, std::shared_ptr<xacc::Observable> H) {
  qrt_impl->exp(q, theta, H);
}

void submit(xacc::AcceleratorBuffer *buffer) { qrt_impl->submit(buffer); }

void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers) {
  qrt_impl->submit(buffers, nBuffers);
}

void set_current_program(std::shared_ptr<xacc::CompositeInstruction> p) {
  qrt_impl->set_current_program(p);
}

void set_current_buffer(xacc::AcceleratorBuffer* buffer) {
  qrt_impl->set_current_buffer(buffer);
}

} // namespace quantum