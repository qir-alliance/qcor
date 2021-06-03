#include "qrt.hpp"

#include <Eigen/Dense>
#include <Utils.hpp>

#include "Instruction.hpp"
#include "PauliOperator.hpp"
#include "pass_manager.hpp"
#include "qcor_config.hpp"
#include "xacc.hpp"
#include "xacc_config.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

namespace xacc {
namespace internal_compiler {
// Extern vars:
int __opt_level = 0;
bool __print_opt_stats = false;
std::string __user_opt_passes = "";
std::string __placement_name = "";
std::vector<int> __qubit_map = {};
std::string __qrt_env = "nisq";
bool __print_final_submission = false;

void execute_pass_manager(
    std::shared_ptr<CompositeInstruction> optional_composite) {
  qcor::internal::PassManager passManager(__opt_level, __qubit_map,
                                          __placement_name);
  auto kernelToExecute = optional_composite
                             ? optional_composite
                             : ::quantum::qrt_impl->get_current_program();
  auto optData = passManager.optimize(kernelToExecute);

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
        qcor::internal::PassManager::runPass(user_pass, kernelToExecute));
  }

  if (__print_opt_stats) {
    // Prints out the Optimizer Stats if requested.
    for (const auto &passData : optData) {
      std::cout << passData.toString(false);
    }
  }

  passManager.applyPlacement(kernelToExecute);
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
}  // namespace internal_compiler
}  // namespace xacc
namespace quantum {
int current_shots = 0;
std::shared_ptr<QuantumRuntime> qrt_impl = nullptr;
std::vector<std::string> kernels_in_translation_unit = {};
std::unordered_map<
    std::string, std::pair<std::vector<std::string>, std::vector<std::string>>>
    kernel_signatures_in_translation_unit = {};

void initialize(const std::string qpu_name, const std::string kernel_name) {
  // if XACC_INSTALL_DIR != XACC_ROOT
  // then we need to pass --xacc-root-path XACC_ROOT
  //
  // Example - we are on Rigetti QCS and can't install via sudo
  // so we dpkg -x xacc to a user directory, but deb package
  // expects to be extracted to /usr/local/xacc, and xacc_config.hpp
  // points to that /usr/local/xacc. Therefore ServiceRegistry fails
  // to load plugins and libs, unless we change rootPath.
  std::string xacc_config_install_dir = std::string(XACC_INSTALL_DIR);
  std::string qcor_config_xacc_root = std::string(XACC_ROOT);
  if (xacc_config_install_dir != qcor_config_xacc_root) {
    std::vector<std::string> cmd_line{"--xacc-root-path",
                                      qcor_config_xacc_root};
    xacc::internal_compiler::compiler_InitializeXACC(qpu_name.c_str(),
                                                     cmd_line);
  } else {
    xacc::internal_compiler::compiler_InitializeXACC(qpu_name.c_str());
  }

  qrt_impl = xacc::getService<QuantumRuntime>(__qrt_env);
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

void set_qrt(const std::string &qrt_name) {
  xacc::internal_compiler::__qrt_env = qrt_name;
}

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

void reset(const qubit &qidx) { qrt_impl->reset(qidx); }

bool mz(const qubit &qidx) { return qrt_impl->mz(qidx); }

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

void set_current_buffer(xacc::AcceleratorBuffer *buffer) {
  qrt_impl->set_current_buffer(buffer);
}

void persistBitstring(xacc::AcceleratorBuffer *buffer) {
  const auto bitstring = buffer->single_measurements_to_bitstring();
  if (!bitstring.empty()) {
    buffer->appendMeasurement(bitstring);
  }
}

void h(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    h(q[i]);
  }
}

void x(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    x(q[i]);
  }
}
void y(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    y(q[i]);
  }
}
void z(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    z(q[i]);
  }
}
void t(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    t(q[i]);
  }
}
void tdg(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    tdg(q[i]);
  }
}
void s(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    s(q[i]);
  }
}
void sdg(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    sdg(q[i]);
  }
}
void mz(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    mz(q[i]);
  }
}

void rx(qreg q, const double theta) {
  for (int i = 0; i < q.size(); i++) {
    rx(q[i], theta);
  }
}
void ry(qreg q, const double theta) {
  for (int i = 0; i < q.size(); i++) {
    ry(q[i], theta);
  }
}
void rz(qreg q, const double theta) {
  for (int i = 0; i < q.size(); i++) {
    rz(q[i], theta);
  }
}
// U1(theta) gate
void u1(qreg q, const double theta) {
  for (int i = 0; i < q.size(); i++) {
    u1(q[i], theta);
  }
}
void u3(qreg q, const double theta, const double phi, const double lambda) {
  for (int i = 0; i < q.size(); i++) {
    u3(q[i], theta, phi, lambda);
  }
}
void reset(qreg q) {
  for (int i = 0; i < q.size(); i++) {
    reset(q[i]);
  }
}

void cnot(qreg src, qreg tgt) {
  assert(src.size() == tgt.size() &&
         "2-qubit broadcast must be across registers of same size.");

  for (int i = 0; i < src.size(); i++) {
    cnot(src[i], tgt[i]);
  }
}

void cy(qreg src, qreg tgt) {
  assert(src.size() == tgt.size() &&
         "2-qubit broadcast must be across registers of same size.");

  for (int i = 0; i < src.size(); i++) {
    cy(src[i], tgt[i]);
  }
}
void cz(qreg src, qreg tgt) {
  assert(src.size() == tgt.size() &&
         "2-qubit broadcast must be across registers of same size.");

  for (int i = 0; i < src.size(); i++) {
    cz(src[i], tgt[i]);
  }
}
void ch(qreg src, qreg tgt) {
  assert(src.size() == tgt.size() &&
         "2-qubit broadcast must be across registers of same size.");

  for (int i = 0; i < src.size(); i++) {
    ch(src[i], tgt[i]);
  }
}

QubitAllocator *getAncillaQubitAllocator() {
  return qrt_impl->get_anc_qubit_allocator();
}
}  // namespace quantum

namespace qcor {
void AncQubitAllocator::onDealloc(xacc::internal_compiler::qubit *in_qubit) {
  // std::cout << "Deallocate: " << (void *)in_qubit << "\n";
  // If this qubit was allocated from this pool:
  if (xacc::container::contains(m_allocatedQubits, in_qubit)) {
    const auto qIndex = std::find(m_allocatedQubits.begin(),
                                  m_allocatedQubits.end(), in_qubit) -
                        m_allocatedQubits.begin();
    // Strategy: create a storage copy of the returned qubit:
    // i.e. with the same index w.r.t. this global anc. buffer
    // but store it in the pool vector -> will stay alive
    // until giving out at the next allocate()
    qubit archive_qubit(ANC_BUFFER_NAME, qIndex, m_buffer.get());
    m_allocatedQubits[qIndex] = &archive_qubit;
    m_qubitPool.emplace_back(archive_qubit);
  }
}

xacc::internal_compiler::qubit AncQubitAllocator::allocate() {
  if (!m_qubitPool.empty()) {
    auto recycled_qubit = m_qubitPool.back();
    m_qubitPool.pop_back();
    return recycled_qubit;
  }
  if (!m_buffer) {
    // This must be the first call.
    assert(m_allocatedQubits.empty());
    m_buffer = xacc::qalloc(1);
    m_buffer->setName(ANC_BUFFER_NAME);
  }

  // Need to allocate new qubit:
  // Each new qubit will have an incrementing index.
  const auto newIdx = m_allocatedQubits.size();
  qubit new_qubit(ANC_BUFFER_NAME, newIdx, m_buffer.get());
  // Just track that we allocated this qubit
  m_allocatedQubits.emplace_back(&new_qubit);
  m_buffer->setSize(m_allocatedQubits.size());
  return new_qubit;
}
} // namespace qcor