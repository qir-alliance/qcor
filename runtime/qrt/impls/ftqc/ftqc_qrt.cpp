
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

namespace qcor {
class FTQC : public quantum::QuantumRuntime {
 public:
  virtual void initialize(const std::string kernel_name) override {
    provider = xacc::getIRProvider("quantum");
    qpu = xacc::internal_compiler::qpu;
  }

  const std::string name() const override { return "ftqc"; }
  const std::string description() const override { return ""; }

  virtual void h(const qubit &qidx) override { applyGate("H", {qidx.second}); }
  virtual void x(const qubit &qidx) override { applyGate("X", {qidx.second}); }
  virtual void y(const qubit &qidx) override { applyGate("Y", {qidx.second}); }
  virtual void z(const qubit &qidx) override { applyGate("Z", {qidx.second}); }
  virtual void t(const qubit &qidx) override { applyGate("T", {qidx.second}); }
  virtual void tdg(const qubit &qidx) override {
    applyGate("Tdg", {qidx.second});
  }
  virtual void s(const qubit &qidx) override { applyGate("S", {qidx.second}); }
  virtual void sdg(const qubit &qidx) override {
    applyGate("Sdg", {qidx.second});
  }

  // Common single-qubit, parameterized instructions
  virtual void rx(const qubit &qidx, const double theta) override {
    applyGate("Rx", {qidx.second}, {theta});
  }
  virtual void ry(const qubit &qidx, const double theta) override {
    applyGate("Ry", {qidx.second}, {theta});
  }
  virtual void rz(const qubit &qidx, const double theta) override {
    applyGate("Rz", {qidx.second}, {theta});
  }
  // U1(theta) gate
  virtual void u1(const qubit &qidx, const double theta) override {
    applyGate("U1", {qidx.second}, {theta});
  }
  virtual void u3(const qubit &qidx, const double theta, const double phi,
                  const double lambda) override {
    applyGate("U", {qidx.second}, {theta, phi, lambda});
  }

  virtual void reset(const qubit &qidx) override {
    applyGate("Reset", {qidx.second});
  }

  // Measure-Z
  virtual bool mz(const qubit &qidx) override {
    applyGate("Measure", {qidx.second});
    // Return the measure result stored in the q reg.
    return (*qReg)[qidx.second];
  }

  // Common two-qubit gates.
  virtual void cnot(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CNOT", {src_idx.second, tgt_idx.second});
  }
  virtual void cy(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CY", {src_idx.second, tgt_idx.second});
  }
  virtual void cz(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CZ", {src_idx.second, tgt_idx.second});
  }
  virtual void ch(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CH", {src_idx.second, tgt_idx.second});
  }
  virtual void swap(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("Swap", {src_idx.second, tgt_idx.second});
  }

  // Common parameterized 2 qubit gates.
  virtual void cphase(const qubit &src_idx, const qubit &tgt_idx,
                      const double theta) override {
    applyGate("CPhase", {src_idx.second, tgt_idx.second}, {theta});
  }
  virtual void crz(const qubit &src_idx, const qubit &tgt_idx,
                   const double theta) override {
    applyGate("CRZ", {src_idx.second, tgt_idx.second}, {theta});
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
  virtual void set_current_program(
      std::shared_ptr<xacc::CompositeInstruction> p) override {
    // Nothing to do
  }
  virtual std::shared_ptr<xacc::CompositeInstruction> get_current_program()
      override {
    return nullptr;
  }

  void set_current_buffer(xacc::AcceleratorBuffer *buffer) override {
    qReg = xacc::as_shared_ptr(buffer);
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
    qpu->apply(qReg, gateInst);
  }

 private:
  std::shared_ptr<xacc::IRProvider> provider;
  std::shared_ptr<xacc::Accelerator> qpu;
  // TODO: eventually, we may want to support an arbitrary number of qubit
  // registers when the FTQC backend can support it.
  std::shared_ptr<xacc::AcceleratorBuffer> qReg;
};
}  // namespace qcor

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
}  // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(FtqcQRTActivator)
