#ifndef RUNTIME_QCOR_QRT_HPP_
#define RUNTIME_QCOR_QRT_HPP_

#include "qalloc.hpp"
#include <CompositeInstruction.hpp>
#include <memory>

#include "AllGateVisitor.hpp"
#include "InstructionIterator.hpp"

using namespace xacc::internal_compiler;

namespace xacc {
class AcceleratorBuffer;
class CompositeInstruction;
class IRProvider;
class Observable;
} // namespace xacc

namespace quantum {

// This represents the public API for the xacc-enabled
// qcor quantum runtime library. The goal here is to provide
// and API that compilers can translate high-level
// quantum kernel language representations to
// (written in openqasm, quil, xasm, etc). The implementation of this API
// seeks to build up an xacc::CompositeInstruction with each individual
// quantum instruction invocation. Once done, clients can invoke
// the submit method to launch the built up program the
// user specified backend.

// Clients must invoke initialize before building up the CompositeInstruction.
// This call will take the name of the backend and the name of the
// CompositeInstruction.

// Note the qubit type is a typedef from xacc, falls back to a
// std::pair<std::string, std::size_t> where the string represents
// the variable name of the qubit register the qubit belongs to, and the
// size_t represents the qubit index in that register.

extern std::shared_ptr<xacc::CompositeInstruction> program;
extern std::shared_ptr<xacc::IRProvider> provider;
extern std::vector<std::string> kernels_in_translation_unit;

void initialize(const std::string qpu_name, const std::string kernel_name);
void set_shots(int shots);
void set_backend(std::string accelerator_name);
void set_backend(std::string accelerator_name, const int shots);
void one_qubit_inst(const std::string &name, const qubit &qidx,
                    std::vector<double> parameters = {});
void two_qubit_inst(const std::string &name, const qubit &qidx1,
                    const qubit &qidx2, std::vector<double> parameters = {});

// Common single-qubit gates.
void h(const qubit &qidx);
void x(const qubit &qidx);
void y(const qubit &qidx);
void z(const qubit &qidx);
void t(const qubit &qidx);
void tdg(const qubit &qidx);
void s(const qubit &qidx);
void sdg(const qubit &qidx);

// Common single-qubit, parameterized instructions
void rx(const qubit &qidx, const double theta);
void ry(const qubit &qidx, const double theta);
void rz(const qubit &qidx, const double theta);
// U1(theta) gate
void u1(const qubit &qidx, const double theta);
void u3(const qubit &qidx, const double theta, const double phi,
        const double lambda);

// Measure-Z
void mz(const qubit &qidx);

// Common two-qubit gates.
void cnot(const qubit &src_idx, const qubit &tgt_idx);
void cy(const qubit &src_idx, const qubit &tgt_idx);
void cz(const qubit &src_idx, const qubit &tgt_idx);
void ch(const qubit &src_idx, const qubit &tgt_idx);
void swap(const qubit &src_idx, const qubit &tgt_idx);

// Common parameterized 2 qubit gates.
void cphase(const qubit &src_idx, const qubit &tgt_idx, const double theta);
void crz(const qubit &src_idx, const qubit &tgt_idx, const double theta);

// exponential of i * theta * H, where H is an Observable pointer
void exp(qreg q, const double theta, xacc::Observable &H);
void exp(qreg q, const double theta, xacc::Observable *H);
void exp(qreg q, const double theta, std::shared_ptr<xacc::Observable> H);

// Submission API. Submit the constructed CompositeInstruction operating
// on the provided AcceleratorBuffer(s) (note qreg wraps an AcceleratorBuffer)
void submit(xacc::AcceleratorBuffer *buffer);
void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers);

// Some getters for the qcor runtime library. 
void set_current_program(std::shared_ptr<xacc::CompositeInstruction> p);

std::shared_ptr<xacc::CompositeInstruction> getProgram();
xacc::CompositeInstruction *program_raw_pointer();

// Clear the current program
void clearProgram();

} // namespace quantum

namespace xacc {

namespace internal_compiler {
// Current runtime controlled bit indices
// (if the current kernel is wrapped in a Controlled block)
extern std::vector<int> __controlledIdx;
// Optimization level: parsed from command line input.
// Convention:
// 0 : no optimization
// 1 : standard optimization (within reasonable walltime limit)
// 2 : extensive optimization (TBD)
extern int __opt_level;
// Should we print out the circuit optimizer stats.
// Disabled by default. Enabled by qcor CLI option.
extern bool __print_opt_stats;

// User-customized passes to run
extern std::string __user_opt_passes;

void simplified_qrt_call_one_qbit(const char *gate_name,
                                  const char *buffer_name,
                                  const std::size_t idx);
void simplified_qrt_call_one_qbit_one_param(const char *gate_name,
                                            const char *buffer_name,
                                            const std::size_t idx,
                                            const double parameter);
void simplified_qrt_call_two_qbits(const char *gate_name,
                                   const char *buffer_name_1,
                                   const char *buffer_name_2,
                                   const std::size_t src_idx,
                                   const std::size_t tgt_idx);
void execute_pass_manager();

class xacc_to_qrt_mapper
    : public xacc::quantum::AllGateVisitor,
      public xacc::InstructionVisitor<xacc::quantum::Circuit> {
protected:
  std::string &buffer_name_to_measure;

public:
  // Ctor: cache the kernel name of the CompositeInstruction
  xacc_to_qrt_mapper(std::string &b) : buffer_name_to_measure(b) {}

  // One-qubit gates
  void visit(xacc::quantum::Hadamard &h) override {
    ::quantum::h({h.getBufferNames()[0], h.bits()[0]});
  }
  void visit(xacc::quantum::Rz &rz) override {
    ::quantum::rz({rz.getBufferNames()[0], rz.bits()[0]},
                  rz.getParameter(0).as<double>());
  }
  void visit(xacc::quantum::Ry &ry) override {
    ::quantum::ry({ry.getBufferNames()[0], ry.bits()[0]},
                  ry.getParameter(0).as<double>());
  }
  void visit(xacc::quantum::Rx &rx) override {
    ::quantum::rx({rx.getBufferNames()[0], rx.bits()[0]},
                  rx.getParameter(0).as<double>());
  }
  void visit(xacc::quantum::X &x) override {
    ::quantum::x({x.getBufferNames()[0], x.bits()[0]});
  }
  void visit(xacc::quantum::Y &y) override {
    ::quantum::y({y.getBufferNames()[0], y.bits()[0]});
  }
  void visit(xacc::quantum::Z &z) override {
    ::quantum::z({z.getBufferNames()[0], z.bits()[0]});
  }
  void visit(xacc::quantum::S &s) override {
    ::quantum::s({s.getBufferNames()[0], s.bits()[0]});
  }
  void visit(xacc::quantum::Sdg &sdg) override {
    ::quantum::sdg({sdg.getBufferNames()[0], sdg.bits()[0]});
  }
  void visit(xacc::quantum::T &t) override {
    ::quantum::t({t.getBufferNames()[0], t.bits()[0]});
  }
  void visit(xacc::quantum::Tdg &tdg) override {
    ::quantum::tdg({tdg.getBufferNames()[0], tdg.bits()[0]});
  }

  // Two-qubit gates
  void visit(xacc::quantum::CNOT &cnot) override {
    ::quantum::cnot({cnot.getBufferNames()[0], cnot.bits()[0]},
                    {cnot.getBufferNames()[1], cnot.bits()[1]});
  };
  void visit(xacc::quantum::CY &cy) override {
    ::quantum::cy({cy.getBufferNames()[0], cy.bits()[0]},
                  {cy.getBufferNames()[1], cy.bits()[1]});
  }
  void visit(xacc::quantum::CZ &cz) override {
    ::quantum::cz({cz.getBufferNames()[0], cz.bits()[0]},
                  {cz.getBufferNames()[1], cz.bits()[1]});
  }
  void visit(xacc::quantum::Swap &s) override {
    ::quantum::swap({s.getBufferNames()[0], s.bits()[0]},
                    {s.getBufferNames()[1], s.bits()[1]});
  }
  void visit(xacc::quantum::CRZ &crz) override {
    ::quantum::crz({crz.getBufferNames()[0], crz.bits()[0]},
                   {crz.getBufferNames()[1], crz.bits()[1]},
                   crz.getParameter(0).as<double>());
  }
  void visit(xacc::quantum::CH &ch) override {
    ::quantum::ch({ch.getBufferNames()[0], ch.bits()[0]},
                  {ch.getBufferNames()[1], ch.bits()[1]});
  }
  void visit(xacc::quantum::CPhase &cphase) override {
    ::quantum::cphase({cphase.getBufferNames()[0], cphase.bits()[0]},
                      {cphase.getBufferNames()[1], cphase.bits()[1]},
                      cphase.getParameter(0).as<double>());
  }

  void visit(xacc::quantum::Measure &measure) override {
    ::quantum::mz({buffer_name_to_measure, measure.bits()[0]});
  }
  void visit(xacc::quantum::Identity &i) override {}
  void visit(xacc::quantum::U &u) override {}
  void visit(xacc::quantum::U1 &u1) override {
    ::quantum::u1({u1.getBufferNames()[0], u1.bits()[0]},
                  u1.getParameter(0).as<double>());
  }
  void visit(xacc::quantum::Circuit &circ) override {}
};

} // namespace internal_compiler
} // namespace xacc

#endif