
#include <Eigen/Dense>
#include <Utils.hpp>

#include "PauliOperator.hpp"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"
using namespace cppmicroservices;

namespace {
class FtqcQubitAllocator : public qcor::AncQubitAllocator {
public:
  static FtqcQubitAllocator *getInstance() {
    if (!g_instance) {
      g_instance = new FtqcQubitAllocator();
    }
    return g_instance;
  }
  static FtqcQubitAllocator *g_instance;
};

FtqcQubitAllocator *FtqcQubitAllocator::g_instance = nullptr;
} // namespace

namespace qcor {
class FTQC : public quantum::QuantumRuntime {
public:
  virtual void initialize(const std::string kernel_name) override {
    provider = xacc::getIRProvider("quantum");
    qpu = xacc::internal_compiler::qpu;
    qubitIdToGlobalIdx.clear();
    setGlobalQubitManager(FtqcQubitAllocator::getInstance());
  }

  void __begin_mark_segment_as_compute() override { mark_as_compute = true; }
  void __end_mark_segment_as_compute() override { mark_as_compute = false; }
  bool isComputeSection() override { return mark_as_compute; }
  const std::string name() const override { return "ftqc"; }
  const std::string description() const override { return ""; }

  virtual void h(const qubit &qidx) override { applyGate("H", {qidx}); }
  virtual void x(const qubit &qidx) override { applyGate("X", {qidx}); }
  virtual void y(const qubit &qidx) override { applyGate("Y", {qidx}); }
  virtual void z(const qubit &qidx) override { applyGate("Z", {qidx}); }
  virtual void t(const qubit &qidx) override { applyGate("T", {qidx}); }
  virtual void tdg(const qubit &qidx) override { applyGate("Tdg", {qidx}); }
  virtual void s(const qubit &qidx) override { applyGate("S", {qidx}); }
  virtual void sdg(const qubit &qidx) override { applyGate("Sdg", {qidx}); }

  // Common single-qubit, parameterized instructions
  virtual void rx(const qubit &qidx, const double theta) override {
    applyGate("Rx", {qidx}, {theta});
  }
  virtual void ry(const qubit &qidx, const double theta) override {
    applyGate("Ry", {qidx}, {theta});
  }
  virtual void rz(const qubit &qidx, const double theta) override {
    applyGate("Rz", {qidx}, {theta});
  }
  // U1(theta) gate
  virtual void u1(const qubit &qidx, const double theta) override {
    applyGate("U1", {qidx}, {theta});
  }
  virtual void u3(const qubit &qidx, const double theta, const double phi,
                  const double lambda) override {
    applyGate("U", {qidx}, {theta, phi, lambda});
  }

  virtual void reset(const qubit &qidx) override { applyGate("Reset", {qidx}); }

  // Measure-Z
  virtual bool mz(const qubit &qidx) override {
    applyGate("Measure", {qidx});
    // Return the measure result stored in the q reg.
    return (*qReg)[qidx.second];
  }

  // Common two-qubit gates.
  virtual void cnot(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CNOT", {src_idx, tgt_idx});
  }
  virtual void cy(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CY", {src_idx, tgt_idx});
  }
  virtual void cz(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CZ", {src_idx, tgt_idx});
  }
  virtual void ch(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CH", {src_idx, tgt_idx});
  }
  virtual void swap(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("Swap", {src_idx, tgt_idx});
  }

  // Common parameterized 2 qubit gates.
  virtual void cphase(const qubit &src_idx, const qubit &tgt_idx,
                      const double theta) override {
    applyGate("CPhase", {src_idx, tgt_idx}, {theta});
  }
  virtual void crz(const qubit &src_idx, const qubit &tgt_idx,
                   const double theta) override {
    applyGate("CRZ", {src_idx, tgt_idx}, {theta});
  }

  // exponential of i * theta * H, where H is an Observable pointer
  virtual void exp(qreg q, const double theta,
                   xacc::Observable &H) override { /* TODO */
  }
  virtual void exp(qreg q, const double theta,
                   xacc::Observable *H) override { /* TODO */
  }
  virtual void exp(qreg q, const double theta,
                   std::shared_ptr<xacc::Observable> H) override { /* TODO */
  }

  // Submission API: sanity check that we don't call these API's.
  // e.g. catch high-level code gen errors.
  virtual void submit(xacc::AcceleratorBuffer *buffer) override {
    throw std::runtime_error("FTQC runtime doesn't support submit API.");
  }
  virtual void submit(xacc::AcceleratorBuffer **buffers,
                      const int nBuffers) override {
    throw std::runtime_error("FTQC runtime doesn't support submit API.");
  }

  void general_instruction(std::shared_ptr<xacc::Instruction> inst) override {
    std::vector<double> params;
    for (auto p : inst->getParameters()) {
      params.push_back(p.as<double>());
    }
    applyGate(inst->name(), inst->bits(), params);
  }

  // Some getters for the qcor runtime library.
  virtual void
  set_current_program(std::shared_ptr<xacc::CompositeInstruction> p) override {
    if (!entryPoint) {
      entryPoint = p;
    } else {
      if (p.get() == entryPoint.get()) {
        // std::cout << "Restart FTQC execution\n";
        instruction_collect_mode = false;
        // Now apply these gates:
        // std::cout << entryPoint->toString() << "\n";
        for (auto &inst : p->getInstructions()) {
          std::vector<size_t> mapped_bits;
          for (int i = 0; i < inst->bits().size(); ++i) {
            const auto qubitId =
                std::make_pair(inst->getBufferNames()[i], inst->bits()[i]);
            if (qubitIdToGlobalIdx.find(qubitId) == qubitIdToGlobalIdx.end()) {
              qubitIdToGlobalIdx[qubitId] = qubitIdToGlobalIdx.size();
              std::stringstream logss;
              logss << "Map " << inst->getBufferNames()[i] << "["
                    << inst->bits()[i] << "] to global ID "
                    << qubitIdToGlobalIdx[qubitId];
              xacc::info(logss.str());
              qReg->setSize(qubitIdToGlobalIdx.size());
            }
            mapped_bits.emplace_back(qubitIdToGlobalIdx[qubitId]);
          }
          inst->setBits(mapped_bits);
          if (__print_final_submission) {
            std::cout << "Apply gate: " << inst->toString() << "\n";
          }
          qpu->apply(qReg, inst);
        }
        // We have executed all pending instructions...
        entryPoint->clear();
      } else {
        // std::cout << "Begin instruction collection\n";
        // Switch to NISQ mode to collect these instructions
        instruction_collector = p;
        instruction_collect_mode = true;
      }
    }
  }
  virtual std::shared_ptr<xacc::CompositeInstruction>
  get_current_program() override {
    return nullptr;
  }

  void set_current_buffer(xacc::AcceleratorBuffer *buffer) override {
    if (!qReg) {
      qReg = xacc::as_shared_ptr(buffer);
      qubitIdToGlobalIdx.clear();
      // The base qreg will always have exact address in the global register.
      for (size_t i = 0; i < buffer->size(); ++i) {
        qubitIdToGlobalIdx[std::make_pair(buffer->name(), i)] = i;
      }
    }
  }

  QubitAllocator *get_anc_qubit_allocator() {
    return FtqcQubitAllocator::getInstance();
  }

private:
  // Notes: all gate parameters must be resolved (to double) for FT-QRT
  // execution.
  void applyGate(const std::string &gateName, const std::vector<size_t> &bits,
                 const std::vector<double> &params = {}) {
    std::vector<xacc::InstructionParameter> instParams;
    for (const auto &val : params) {
      instParams.emplace_back(val);
    }
    auto gateInst = provider->createInstruction(gateName, bits, instParams);
    if (mark_as_compute) {
      gateInst->attachMetadata({{"__qcor__compute__segment__", true}});
    }
    qpu->apply(qReg, gateInst);
  }

  void applyGate(const std::string &gateName,
                 std::initializer_list<size_t> bits,
                 const std::vector<double> &params = {}) {
    applyGate(gateName, std::vector<size_t>(bits), params);
  }

  void applyGate(const std::string &gateName, const std::vector<qubit> &qbits,
                 const std::vector<double> &params = {}) {
    std::vector<xacc::InstructionParameter> instParams;
    for (const auto &val : params) {
      instParams.emplace_back(val);
    }
    std::vector<size_t> bits;
    for (const auto &qb : qbits) {
      // Never seen this qubit
      const auto qubitId = std::make_pair(qb.first, qb.second);
      if (qubitIdToGlobalIdx.find(qubitId) == qubitIdToGlobalIdx.end()) {
        qubitIdToGlobalIdx[qubitId] = qubitIdToGlobalIdx.size();
        std::stringstream logss;
        logss << "Map " << qb.first << "[" << qb.second << "] to global ID "
              << qubitIdToGlobalIdx[qubitId];
        xacc::info(logss.str());
        qReg->setSize(qubitIdToGlobalIdx.size());
      }
      bits.emplace_back(qubitIdToGlobalIdx[qubitId]);
    }
    auto gateInst = provider->createInstruction(gateName, bits, instParams);
    if (mark_as_compute) {
      gateInst->attachMetadata({{"__qcor__compute__segment__", true}});
    }

    if (!instruction_collect_mode) {
      if (__print_final_submission) {
        std::cout << "Apply gate: " << gateInst->toString() << "\n";
      }
      qpu->apply(qReg, gateInst);
    } else {
      // Note: for instruction collection, we must keep the
      // buffer names as-is (only map to global reg when executed)
      std::vector<std::string> buffer_names;
      std::vector<size_t> bits;
      for (const auto &qb : qbits) {
        bits.emplace_back(qb.second);
        buffer_names.emplace_back(qb.first);
      }
      gateInst->setBits(bits);
      gateInst->setBufferNames(buffer_names);
      instruction_collector->addInstruction(gateInst);
    }
  }

private:
  bool mark_as_compute = false;
  // Are we in a instruction collection mode?
  // cannot execute the instructions now.
  bool instruction_collect_mode = false;
  std::shared_ptr<xacc::CompositeInstruction> instruction_collector;
  std::shared_ptr<xacc::IRProvider> provider;
  std::shared_ptr<xacc::Accelerator> qpu;
  // TODO: eventually, we may want to support an arbitrary number of qubit
  // registers when the FTQC backend can support it.
  std::shared_ptr<xacc::AcceleratorBuffer> qReg;
  std::map<std::pair<std::string, size_t>, size_t> qubitIdToGlobalIdx;
  std::shared_ptr<xacc::CompositeInstruction> entryPoint;
};
} // namespace qcor

namespace {
class US_ABI_LOCAL FtqcQRTActivator : public BundleActivator {
public:
  FtqcQRTActivator() {}
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::FTQC>();
    context.RegisterService<quantum::QuantumRuntime>(xt);
  }
  void Stop(BundleContext /*context*/) {}
};
} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(FtqcQRTActivator)
