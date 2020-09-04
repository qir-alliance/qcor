
#include "PauliOperator.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"
#include <Eigen/Dense>
#include <Utils.hpp>

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
using namespace cppmicroservices;

namespace qcor {
class FTQC : public quantum::QuantumRuntime {
public:
  virtual void initialize(const std::string kernel_name) override { /* TODO */
  }

  const std::string name() const override { return "ftqc"; }
  const std::string description() const override { return ""; }

  virtual void h(const qubit &qidx) override { /* TODO */
  }
  virtual void x(const qubit &qidx) override { /* TODO */
  }
  virtual void y(const qubit &qidx) override { /* TODO */
  }
  virtual void z(const qubit &qidx) override { /* TODO */
  }
  virtual void t(const qubit &qidx) override { /* TODO */
  }
  virtual void tdg(const qubit &qidx) override { /* TODO */
  }
  virtual void s(const qubit &qidx) override { /* TODO */
  }
  virtual void sdg(const qubit &qidx) override { /* TODO */
  }

  // Common single-qubit, parameterized instructions
  virtual void rx(const qubit &qidx, const double theta) override { /* TODO */
  }
  virtual void ry(const qubit &qidx, const double theta) override { /* TODO */
  }
  virtual void rz(const qubit &qidx, const double theta) override { /* TODO */
  }
  // U1(theta) gate
  virtual void u1(const qubit &qidx, const double theta) override { /* TODO */
  }
  virtual void u3(const qubit &qidx, const double theta, const double phi,
                  const double lambda) override { /* TODO */
  }

  // Measure-Z
  virtual void mz(const qubit &qidx) override { /* TODO */
  }

  // Common two-qubit gates.
  virtual void cnot(const qubit &src_idx,
                    const qubit &tgt_idx) override { /* TODO */
  }
  virtual void cy(const qubit &src_idx,
                  const qubit &tgt_idx) override { /* TODO */
  }
  virtual void cz(const qubit &src_idx,
                  const qubit &tgt_idx) override { /* TODO */
  }
  virtual void ch(const qubit &src_idx,
                  const qubit &tgt_idx) override { /* TODO */
  }
  virtual void swap(const qubit &src_idx,
                    const qubit &tgt_idx) override { /* TODO */
  }

  // Common parameterized 2 qubit gates.
  virtual void cphase(const qubit &src_idx, const qubit &tgt_idx,
                      const double theta) override { /* TODO */
  }
  virtual void crz(const qubit &src_idx, const qubit &tgt_idx,
                   const double theta) override { /* TODO */
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

  // Submission API. Submit the constructed CompositeInstruction operating
  // on the provided AcceleratorBuffer(s) (note qreg wraps an AcceleratorBuffer)
  virtual void submit(xacc::AcceleratorBuffer *buffer) override { /* TODO */
  }
  virtual void submit(xacc::AcceleratorBuffer **buffers,
                      const int nBuffers) override { /* TODO */
  }

  // Some getters for the qcor runtime library.
  virtual void set_current_program(
      std::shared_ptr<xacc::CompositeInstruction> p) override { /* TODO */
  }
  virtual std::shared_ptr<xacc::CompositeInstruction>
  get_current_program() override { /* TODO */
    return nullptr;
  }
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
