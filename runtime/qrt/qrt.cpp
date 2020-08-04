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
std::vector<int> __controlledIdx = {};

void simplified_qrt_call_one_qbit(const char *gate_name,
                                  const char *buffer_name,
                                  const std::size_t idx) {
  ::quantum::one_qubit_inst(gate_name, {buffer_name, idx});
}

void simplified_qrt_call_one_qbit_one_param(const char *gate_name,
                                            const char *buffer_name,
                                            const std::size_t idx,
                                            const double parameter) {
  ::quantum::one_qubit_inst(gate_name, {buffer_name, idx}, {parameter});
}

void simplified_qrt_call_two_qbits(const char *gate_name,
                                   const char *buffer_name_1,
                                   const char *buffer_name_2,
                                   const std::size_t src_idx,
                                   const std::size_t tgt_idx) {
  ::quantum::two_qubit_inst(gate_name, {buffer_name_1, src_idx},
                            {buffer_name_1, tgt_idx});
}

void execute_pass_manager() {
  qcor::internal::PassManager passManager(__opt_level);
  const auto optData = passManager.optimize(::quantum::program);
  if (__print_opt_stats) {
    // Prints out the Optimizer Stats if requested.
    for (const auto &passData : optData) {
      std::cout << passData.toString(false);
    }
  }
}
void execute_pass(const char *passName) {
  const auto passData = qcor::internal::PassManager::runPass(passName, ::quantum::program);
  std::cout << passData.toString(false);
}

} // namespace internal_compiler
} // namespace xacc
namespace quantum {
std::shared_ptr<xacc::CompositeInstruction> program = nullptr;
std::shared_ptr<xacc::IRProvider> provider = nullptr;
std::vector<std::string> kernels_in_translation_unit = {};

// We only allow *single* quantum entry point,
// i.e. a master quantum kernel which is invoked from classical code.
// Multiple kernels can be defined to be used inside the *entry-point* kernel.
// Once the *entry-point* kernel has been invoked, initialize() calls
// by sub-kernels will be ignored.
bool __entry_point_initialized = false;

void initialize(const std::string qpu_name, const std::string kernel_name) {
  if (!__entry_point_initialized) {
    xacc::internal_compiler::compiler_InitializeXACC(qpu_name.c_str());
    provider = xacc::getIRProvider("quantum");
    program = provider->createComposite(kernel_name);
  }

  __entry_point_initialized = true;
}

void set_backend(std::string accelerator_name, const int shots) {

  xacc::internal_compiler::compiler_InitializeXACC(accelerator_name.c_str());
  provider = xacc::getIRProvider("quantum");

  set_shots(shots);
}

void set_backend(std::string accelerator_name) {
  xacc::internal_compiler::compiler_InitializeXACC(accelerator_name.c_str());
  provider = xacc::getIRProvider("quantum");
}

void set_shots(int shots) {
  xacc::internal_compiler::get_qpu()->updateConfiguration(
      {std::make_pair("shots", shots)});
}

// Add a controlled instruction:
void add_controlled_inst(xacc::InstPtr &inst, int ctrlIdx) {
  auto tempKernel = provider->createComposite("temp_control");
  tempKernel->addInstruction(inst);
  auto ctrlKernel = std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      xacc::getService<xacc::Instruction>("C-U"));
  ctrlKernel->expand({
      std::make_pair("U", tempKernel),
      std::make_pair("control-idx", ctrlIdx),
  });

  for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
    program->addInstruction(ctrlKernel->getInstruction(instId)->clone());
  }
}

void one_qubit_inst(const std::string &name, const qubit &qidx,
                    std::vector<double> parameters) {
  auto inst =
      provider->createInstruction(name, std::vector<std::size_t>{qidx.second});
  inst->setBufferNames({qidx.first});
  for (int i = 0; i < parameters.size(); i++) {
    inst->setParameter(i, parameters[i]);
  }
  // Not in a controlled-block
  if (xacc::internal_compiler::__controlledIdx.empty()) {
    // Add the instruction
    program->addInstruction(inst);
  } else {
    // In a controlled block:
    add_controlled_inst(inst, __controlledIdx[0]);
  }
}

void two_qubit_inst(const std::string &name, const qubit &qidx1,
                    const qubit &qidx2, std::vector<double> parameters) {
  auto inst = provider->createInstruction(
      name, std::vector<std::size_t>{qidx1.second, qidx2.second});
  inst->setBufferNames({qidx1.first, qidx2.first});
  for (int i = 0; i < parameters.size(); i++) {
    inst->setParameter(i, parameters[i]);
  }
  // Not in a controlled-block
  if (xacc::internal_compiler::__controlledIdx.empty()) {
    program->addInstruction(inst);
  } else {
    // In a controlled block:
    add_controlled_inst(inst, __controlledIdx[0]);
  }
}

void h(const qubit &qidx) { one_qubit_inst("H", qidx); }
void x(const qubit &qidx) { one_qubit_inst("X", qidx); }
void y(const qubit &qidx) { one_qubit_inst("Y", qidx); }
void z(const qubit &qidx) { one_qubit_inst("Z", qidx); }

void s(const qubit &qidx) { one_qubit_inst("S", qidx); }
void sdg(const qubit &qidx) { one_qubit_inst("Sdg", qidx); }

void t(const qubit &qidx) { one_qubit_inst("T", qidx); }
void tdg(const qubit &qidx) { one_qubit_inst("Tdg", qidx); }

void rx(const qubit &qidx, const double theta) {
  one_qubit_inst("Rx", qidx, {theta});
}

void ry(const qubit &qidx, const double theta) {
  one_qubit_inst("Ry", qidx, {theta});
}

void rz(const qubit &qidx, const double theta) {
  one_qubit_inst("Rz", qidx, {theta});
}

void u1(const qubit &qidx, const double theta) {
  one_qubit_inst("U1", qidx, {theta});
}

void u3(const qubit &qidx, const double theta, const double phi,
        const double lambda) {
  one_qubit_inst("U", qidx, {theta, phi, lambda});
}

void mz(const qubit &qidx) { one_qubit_inst("Measure", qidx); }

void cnot(const qubit &src_idx, const qubit &tgt_idx) {
  two_qubit_inst("CNOT", src_idx, tgt_idx);
}

void cy(const qubit &src_idx, const qubit &tgt_idx) {
  two_qubit_inst("CY", src_idx, tgt_idx);
}

void cz(const qubit &src_idx, const qubit &tgt_idx) {
  two_qubit_inst("CZ", src_idx, tgt_idx);
}

void ch(const qubit &src_idx, const qubit &tgt_idx) {
  two_qubit_inst("CH", src_idx, tgt_idx);
}

void swap(const qubit &src_idx, const qubit &tgt_idx) {
  two_qubit_inst("Swap", src_idx, tgt_idx);
}

void cphase(const qubit &src_idx, const qubit &tgt_idx, const double theta) {
  two_qubit_inst("CPhase", src_idx, tgt_idx, {theta});
}

void crz(const qubit &src_idx, const qubit &tgt_idx, const double theta) {
  two_qubit_inst("CRZ", src_idx, tgt_idx, {theta});
}

void exp(qreg q, const double theta, xacc::Observable &H) {
  exp(q, theta, xacc::as_shared_ptr(&H));
}

void exp(qreg q, const double theta, xacc::Observable *H) {
  exp(q, theta, xacc::as_shared_ptr(H));
}

void exp(qreg q, const double theta, std::shared_ptr<xacc::Observable> H) {

  std::unordered_map<std::string, xacc::quantum::Term> terms;

  terms = dynamic_cast<xacc::quantum::PauliOperator *>(H.get())->getTerms();

  double pi = xacc::constants::pi;
  auto gateRegistry = xacc::getIRProvider("quantum");
  std::string xasm_src = "";

  for (auto inst : terms) {

    auto spinInst = inst.second;

    // Get the individual pauli terms
    auto termsMap = std::get<2>(spinInst);

    std::vector<std::pair<int, std::string>> terms;
    for (auto &kv : termsMap) {
      if (kv.second != "I" && !kv.second.empty()) {
        terms.push_back({kv.first, kv.second});
      }
    }
    // The largest qubit index is on the last term
    int largestQbitIdx = terms[terms.size() - 1].first;

    std::vector<std::size_t> qidxs;
    std::stringstream basis_front, basis_back;

    for (auto &term : terms) {

      auto qid = term.first;
      auto pop = term.second;

      qidxs.push_back(qid);

      if (pop == "X") {

        basis_front << "H(q[" << qid << "]);\n";
        basis_back << "H(q[" << qid << "]);\n";

      } else if (pop == "Y") {
        basis_front << "Rx(q[" << qid << "], " << 1.57079362679 << ");\n";
        basis_back << "Rx(q[" << qid << "], " << -1.57079362679 << ");\n";
      }
    }

    // std::cout << "QIDS:  " << qidxs << "\n";

    Eigen::MatrixXi cnot_pairs(2, qidxs.size() - 1);
    for (int i = 0; i < qidxs.size() - 1; i++) {
      cnot_pairs(0, i) = qidxs[i];
    }
    for (int i = 0; i < qidxs.size() - 1; i++) {
      cnot_pairs(1, i) = qidxs[i + 1];
    }

    // std::cout << "HOWDY: \n" << cnot_pairs << "\n";
    std::stringstream cnot_front, cnot_back;
    for (int i = 0; i < qidxs.size() - 1; i++) {
      Eigen::VectorXi pairs = cnot_pairs.col(i);
      auto c = pairs(0);
      auto t = pairs(1);
      cnot_front << "CNOT(q[" << c << "], q[" << t << "]);\n";
    }

    for (int i = qidxs.size() - 2; i >= 0; i--) {
      Eigen::VectorXi pairs = cnot_pairs.col(i);
      auto c = pairs(0);
      auto t = pairs(1);
      cnot_back << "CNOT(q[" << c << "], q[" << t << "]);\n";
    }

    xasm_src = xasm_src + "\n" + basis_front.str() + cnot_front.str();

    xasm_src = xasm_src + "Rz(q[" + std::to_string(qidxs[qidxs.size() - 1]) +
               "], " + std::to_string(std::real(spinInst.coeff()) * theta) +
               ");\n";

    xasm_src = xasm_src + cnot_back.str() + basis_back.str();
  }

  int name_counter = 1;
  std::string name = "exp_tmp";
  while (xacc::hasCompiled(name)) {
    name += std::to_string(name_counter);
  }

  xasm_src = "__qpu__ void " + name + "(qbit q) {\n" + xasm_src + "}";

  // std::cout << xasm_src << "\n";
  auto xasm = xacc::getCompiler("xasm");
  auto tmp = xasm->compile(xasm_src)->getComposites()[0];

  for (auto inst : tmp->getInstructions()) {
    program->addInstruction(inst);
  }
}

void submit(xacc::AcceleratorBuffer *buffer) {
  xacc::internal_compiler::execute_pass_manager();
  xacc::internal_compiler::execute(buffer, program);
  clearProgram();
}

void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers) {
  xacc::internal_compiler::execute(buffers, nBuffers, program);
}

void set_current_program(std::shared_ptr<xacc::CompositeInstruction> p) {
  program = p;
}

std::shared_ptr<xacc::CompositeInstruction> getProgram() { return program; }
xacc::CompositeInstruction *program_raw_pointer() { return program.get(); }
void clearProgram() {
  if (program && provider)
    program = provider->createComposite(program->name());
}
} // namespace quantum