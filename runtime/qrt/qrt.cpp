#include "qrt.hpp"
#include "Instruction.hpp"
#include "PauliOperator.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include <Eigen/Dense>
#include <Utils.hpp>

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

void mz(const qubit &qidx) { one_qubit_inst("Measure", qidx); }

void cnot(const qubit &src_idx, const qubit &tgt_idx) {
  auto cx = provider->createInstruction(
      "CNOT", std::vector<std::size_t>{src_idx.second, tgt_idx.second});
  cx->setBufferNames({src_idx.first, tgt_idx.first});
  program->addInstruction(cx);
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
  xacc::internal_compiler::execute(buffer, program);
}

void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers) {
  xacc::internal_compiler::execute(buffers, nBuffers, program);
}
std::shared_ptr<xacc::CompositeInstruction> getProgram() { return program; }
} // namespace quantum