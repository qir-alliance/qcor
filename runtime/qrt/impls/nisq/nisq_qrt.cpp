#include <Eigen/Dense>
#include <Utils.hpp>
#include <fstream>
#include <unistd.h>
#include <stdio.h>

#include "CommonGates.hpp"
#include "FermionOperator.hpp"
#include "ObservableTransform.hpp"
#include "PauliOperator.hpp"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_observable.hpp"
#include "xacc_service.hpp"
#include "qcor_observable.hpp"

using namespace cppmicroservices;
using namespace xacc;

namespace {
class NisqQubitAllocator : public qcor::AncQubitAllocator {
 public:
  static NisqQubitAllocator *getInstance() {
    if (!g_instance) {
      g_instance = new NisqQubitAllocator();
    }
    return g_instance;
  }
  static NisqQubitAllocator *g_instance;
};

NisqQubitAllocator *NisqQubitAllocator::g_instance = nullptr;
}  // namespace

namespace qcor {
template <typename T>
bool ptr_is_a(std::shared_ptr<Observable> ptr) {
  return std::dynamic_pointer_cast<T>(ptr) != nullptr;
}
class NISQ : public ::quantum::QuantumRuntime,
             // Cloneable for use in qir-qrt ctrl, pow, adj regions.
             public xacc::Cloneable<::quantum::QuantumRuntime> {
 protected:
  bool mark_as_compute = false;
  std::shared_ptr<CompositeInstruction> program;
  std::shared_ptr<xacc::IRProvider> provider;

  void one_qubit_inst(const std::string &name, const qubit &qidx,
                      std::vector<double> parameters = {}) {
    auto inst = provider->createInstruction(
        name, std::vector<std::size_t>{qidx.second});
    inst->setBufferNames({qidx.first});
    for (int i = 0; i < parameters.size(); i++) {
      inst->setParameter(i, parameters[i]);
    }

    if (mark_as_compute) {
      inst->attachMetadata({{"__qcor__compute__segment__", true}});
    }

    program->addInstruction(inst);
  }

  void two_qubit_inst(const std::string &name, const qubit &qidx1,
                      const qubit &qidx2, std::vector<double> parameters = {}) {
    auto inst = provider->createInstruction(
        name, std::vector<std::size_t>{qidx1.second, qidx2.second});
    inst->setBufferNames({qidx1.first, qidx2.first});
    for (int i = 0; i < parameters.size(); i++) {
      inst->setParameter(i, parameters[i]);
    }

    if (mark_as_compute) {
      inst->attachMetadata({{"__qcor__compute__segment__", true}});
    }

    program->addInstruction(inst);
  }

 public:
  std::shared_ptr<::quantum::QuantumRuntime> clone() override {
    return std::make_shared<NISQ>();
  }

  void initialize(const std::string kernel_name) override {
    provider = xacc::getIRProvider("quantum");
    program = std::make_shared<CompositeInstruction>(kernel_name);
    setGlobalQubitManager(NisqQubitAllocator::getInstance());
  }

  QubitAllocator *get_anc_qubit_allocator() {
    return NisqQubitAllocator::getInstance();
  }

  void __begin_mark_segment_as_compute() override { mark_as_compute = true; }
  void __end_mark_segment_as_compute() override { mark_as_compute = false; }
  bool isComputeSection() override { return mark_as_compute; }

  void h(const qubit &qidx) override { one_qubit_inst("H", qidx); }
  void x(const qubit &qidx) override { one_qubit_inst("X", qidx); }
  void y(const qubit &qidx) override { one_qubit_inst("Y", qidx); }
  void z(const qubit &qidx) override { one_qubit_inst("Z", qidx); }

  void s(const qubit &qidx) override { one_qubit_inst("S", qidx); }
  void sdg(const qubit &qidx) override { one_qubit_inst("Sdg", qidx); }

  void t(const qubit &qidx) override { one_qubit_inst("T", qidx); }
  void tdg(const qubit &qidx) override { one_qubit_inst("Tdg", qidx); }

  void rx(const qubit &qidx, const double theta) override {
    one_qubit_inst("Rx", qidx, {theta});
  }

  void ry(const qubit &qidx, const double theta) override {
    one_qubit_inst("Ry", qidx, {theta});
  }

  void rz(const qubit &qidx, const double theta) override {
    one_qubit_inst("Rz", qidx, {theta});
  }

  void u1(const qubit &qidx, const double theta) override {
    one_qubit_inst("U1", qidx, {theta});
  }

  void u3(const qubit &qidx, const double theta, const double phi,
          const double lambda) override {
    one_qubit_inst("U", qidx, {theta, phi, lambda});
  }

  void reset(const qubit &qidx) override { one_qubit_inst("Reset", qidx); }

  bool mz(const qubit &qidx) override {
    one_qubit_inst("Measure", qidx);
    return false;
  }

  void cnot(const qubit &src_idx, const qubit &tgt_idx) override {
    two_qubit_inst("CNOT", src_idx, tgt_idx);
  }

  void cy(const qubit &src_idx, const qubit &tgt_idx) override {
    two_qubit_inst("CY", src_idx, tgt_idx);
  }

  void cz(const qubit &src_idx, const qubit &tgt_idx) override {
    two_qubit_inst("CZ", src_idx, tgt_idx);
  }

  void ch(const qubit &src_idx, const qubit &tgt_idx) override {
    two_qubit_inst("CH", src_idx, tgt_idx);
  }

  void swap(const qubit &src_idx, const qubit &tgt_idx) override {
    two_qubit_inst("Swap", src_idx, tgt_idx);
  }

  void cphase(const qubit &src_idx, const qubit &tgt_idx,
              const double theta) override {
    two_qubit_inst("CPhase", src_idx, tgt_idx, {theta});
  }

  void crz(const qubit &src_idx, const qubit &tgt_idx,
           const double theta) override {
    two_qubit_inst("CRZ", src_idx, tgt_idx, {theta});
  }

  void general_instruction(std::shared_ptr<xacc::Instruction> inst) override {
    std::vector<double> params;
    for (auto p : inst->getParameters()) {
      params.push_back(p.as<double>());
    }
    if (inst->bits().size() == 1) {
      one_qubit_inst(
          inst->name(),
          qubit{
              inst->getBufferNames().empty() ? "q" : inst->getBufferNames()[0],
              inst->bits()[0]},
          params);
    } else if (inst->bits().size() == 2) {
      two_qubit_inst(
          inst->name(),
          qubit{
              inst->getBufferNames().empty() ? "q" : inst->getBufferNames()[0],
              inst->bits()[0]},
          qubit{
              inst->getBufferNames().empty() ? "q" : inst->getBufferNames()[1],
              inst->bits()[1]},
          params);
    } else {
      xacc::error(
          "Nisq quantum runtime general_instruction can only take 1 and 2 "
          "qubit operations.");
    }
    return;
  }

  // void exp(qreg q, const double theta, xacc::Observable &H) override {
  //   exp(q, theta, xacc::as_shared_ptr(&H));
  // }

  // void exp(qreg q, const double theta, Operator *H) override {
  //   exp(q, theta, xacc::as_shared_ptr(H));
  // }

  void exp(qreg q, const double theta, Operator &op) override {
    auto Hptr_input =
        std::dynamic_pointer_cast<xacc::Observable>(op.get_as_opaque());

    auto obs_str = Hptr_input->toString();
    auto fermi_to_pauli = xacc::getService<xacc::ObservableTransform>("jw");
    std::shared_ptr<xacc::Observable> Hptr;
    if (ptr_is_a<xacc::quantum::FermionOperator>(Hptr_input)) {
      Hptr = fermi_to_pauli->transform(Hptr_input);
    } else if (obs_str.find("^") != std::string::npos) {
      auto fermionObservable = xacc::quantum::getObservable("fermion", obs_str);
      Hptr = fermi_to_pauli->transform(fermionObservable);
    } else if (ptr_is_a<xacc::quantum::PauliOperator>(Hptr_input)) {
      Hptr = Hptr_input;
    } else if (obs_str.find("X") != std::string::npos ||
               obs_str.find("Y") != std::string::npos ||
               obs_str.find("Z") != std::string::npos) {
      Hptr = xacc::quantum::getObservable("pauli", obs_str);
    } else {
      xacc::error(
          "[qcor::exp()] Error, cannot cast incoming Observable ptr to "
          "something we can process.");
    }

    std::map<std::string, xacc::quantum::Term> terms;

    xacc::ScopeTimer timer("timer", false);
      // Convert the IR into a Hamiltonian
    xacc::quantum::PauliOperator &H =
        dynamic_cast<xacc::quantum::PauliOperator &>(*Hptr.get());
    terms = H.getTerms();

    double pi = xacc::constants::pi;
    auto gateRegistry = xacc::getIRProvider("quantum");
    std::vector<xacc::InstPtr> exp_insts;

    auto q_name = q.name();
    for (auto inst : terms) {
      auto spinInst = inst.second;
      if (spinInst.isIdentity()) {
        continue;
      }
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
      std::vector<xacc::InstPtr> basis_front, basis_back;

      for (auto &term : terms) {
        auto qid = term.first;
        auto pop = term.second;

        qidxs.push_back(qid);

        if (pop == "X") {
          basis_front.emplace_back(
              std::make_shared<xacc::quantum::Hadamard>(q[qid].second));
          basis_back.emplace_back(
              std::make_shared<xacc::quantum::Hadamard>(q[qid].second));

          basis_front.back()->setBufferNames(
              std::vector<std::string>(1, q[qid].first));
          basis_back.back()->setBufferNames(
              std::vector<std::string>(1, q[qid].first));
        } else if (pop == "Y") {
          basis_front.emplace_back(
              std::make_shared<xacc::quantum::Rx>(q[qid].second, 1.57079362679));
          basis_back.emplace_back(
              std::make_shared<xacc::quantum::Rx>(q[qid].second,
              -1.57079362679));

          basis_front.back()->setBufferNames(
              std::vector<std::string>(1, q[qid].first));
          basis_back.back()->setBufferNames(
              std::vector<std::string>(1, q[qid].first));
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
      std::vector<xacc::InstPtr> cnot_front, cnot_back;
      for (int i = 0; i < qidxs.size() - 1; i++) {
        Eigen::VectorXi pairs = cnot_pairs.col(i);
        auto c = pairs(0);
        auto t = pairs(1);
        cnot_front.emplace_back(std::make_shared<xacc::quantum::CNOT>(q[c].second,
        q[t].second));

        cnot_front.back()->setBufferNames(std::vector<std::string>{q[c].first,
        q[t].first});
      }

      for (int i = qidxs.size() - 2; i >= 0; i--) {
        Eigen::VectorXi pairs = cnot_pairs.col(i);
        auto c = pairs(0);
        auto t = pairs(1);
        cnot_back.emplace_back(std::make_shared<xacc::quantum::CNOT>(q[c].second,
        q[t].second));
        cnot_back.back()->setBufferNames(std::vector<std::string>{q[c].first,
        q[t].first});
      }
      exp_insts.insert(exp_insts.end(),
                       std::make_move_iterator(basis_front.begin()),
                       std::make_move_iterator(basis_front.end()));
      exp_insts.insert(exp_insts.end(),
                       std::make_move_iterator(cnot_front.begin()),
                       std::make_move_iterator(cnot_front.end()));

      // FIXME, we assume real coefficients, if its zero,
      // check that the imag part is not zero and use it
      if (std::fabs(std::real(spinInst.coeff())) > 1e-12) {
        exp_insts.emplace_back(std::make_shared<xacc::quantum::Rz>(
            q[qidxs[qidxs.size() - 1]].second, std::real(spinInst.coeff()) *
            theta));
        exp_insts.back()->setBufferNames(std::vector<std::string>(1,
        q[qidxs[qidxs.size() - 1]].first));

      } else if (std::fabs(std::imag(spinInst.coeff())) > 1e-12) {
        exp_insts.emplace_back(std::make_shared<xacc::quantum::Rz>(
            q[qidxs[qidxs.size() - 1]].second, std::imag(spinInst.coeff()) *
            theta));
        exp_insts.back()->setBufferNames(std::vector<std::string>(1,
        q[qidxs[qidxs.size() - 1]].first));
      }
      exp_insts.insert(exp_insts.end(),
                       std::make_move_iterator(cnot_back.begin()),
                       std::make_move_iterator(cnot_back.end()));
      exp_insts.insert(exp_insts.end(),
                       std::make_move_iterator(basis_back.begin()),
                       std::make_move_iterator(basis_back.end()));
    }

    program->addInstructions(std::move(exp_insts), false);
  }

  void submit(xacc::AcceleratorBuffer *buffer) override {
    // xacc::internal_compiler::execute_pass_manager();
    auto anc_allocator = NisqQubitAllocator::getInstance();
    if (anc_allocator->get_buffer() &&
        anc_allocator->get_buffer()->size() > 0) {
      std::vector<xacc::AcceleratorBuffer *> buffer_list{
          buffer, anc_allocator->get_buffer().get()};
      submit(buffer_list.data(), buffer_list.size());
    } else {
      if (__print_final_submission) {
        std::cout << "SUBMIT:\n" << program->toString() << "\n";
        if (!__print_final_submission_filename.empty()) {
          std::ofstream os(__print_final_submission_filename);
          os << program->toString();
          os.close();
        }
      }
      xacc::internal_compiler::execute(
          buffer, program->as_xacc());
    }

    clearProgram();
  }

  void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers) override {
    // What if we get an array of buffers but they
    // are all the same pointer
    std::set<xacc::AcceleratorBuffer *> ptrs;
    for (int i = 0; i < nBuffers; i++) {
      ptrs.insert(buffers[i]);
    }
    // Add the kernel-allocated temporary buffer if necessary:
    auto anc_allocator = NisqQubitAllocator::getInstance();
    if (anc_allocator->get_buffer() &&
        anc_allocator->get_buffer()->size() > 0) {
      ptrs.insert(anc_allocator->get_buffer().get());
    }
    // If size is 1 here, then we only have
    // one pointer, like in the case of qubit.results()
    if (ptrs.size() == 1) {
      submit(buffers[0]);
      return;
    }

    if (__print_final_submission) {
      std::cout << "SUBMIT:\n" << program->toString() << "\n";
      if (!__print_final_submission_filename.empty()) {
        std::ofstream os(__print_final_submission_filename);
        os << program->toString();
        os.close();
      }
    }
    xacc::internal_compiler::execute(buffers, nBuffers, program->as_xacc());
  }

  void set_current_program(std::shared_ptr<CompositeInstruction> p) override {
    program = p;
  }
  std::shared_ptr<CompositeInstruction> get_current_program() override {
    return program;
  }
  void clearProgram() {
    if (program && provider)
      program = std::make_shared<CompositeInstruction>(program->name());
  }

  void set_current_buffer(xacc::AcceleratorBuffer *buffer) override {
    // Nothing to do: the NISQ runtime doesn't keep track of runtime buffer
    // info.
  }

  const std::string name() const override { return "nisq"; }
  const std::string description() const override { return ""; }
};
}  // namespace qcor

namespace {

/**
 */
class US_ABI_LOCAL NisqQRTActivator : public BundleActivator {
 public:
  NisqQRTActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::NISQ>();
    context.RegisterService<::quantum::QuantumRuntime>(xt);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

}  // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(NisqQRTActivator)
