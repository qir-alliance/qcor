#ifndef RUNTIME_QCOR_QRT_HPP_
#define RUNTIME_QCOR_QRT_HPP_

#include "Identifiable.hpp"
#include "qalloc.hpp"
#include <memory>
#include <unordered_map>
#include "qcor_ir.hpp"

namespace qcor {
  class Operator;
}
using namespace xacc::internal_compiler;

namespace xacc {
class AcceleratorBuffer;
// class CompositeInstruction;
class Instruction;
class IRProvider;
class Observable;
} // namespace xacc

namespace quantum {

class QuantumRuntime : public xacc::Identifiable {

public:
  virtual void initialize(const std::string kernel_name) = 0;
  virtual void finalize() {};
  virtual void __begin_mark_segment_as_compute() = 0;
  virtual void __end_mark_segment_as_compute() = 0;
  virtual bool isComputeSection() = 0;
  
  virtual void h(const qubit &qidx) = 0;
  virtual void x(const qubit &qidx) = 0;
  virtual void y(const qubit &qidx) = 0;
  virtual void z(const qubit &qidx) = 0;
  virtual void t(const qubit &qidx) = 0;
  virtual void tdg(const qubit &qidx) = 0;
  virtual void s(const qubit &qidx) = 0;
  virtual void sdg(const qubit &qidx) = 0;

  // Common single-qubit, parameterized instructions
  virtual void rx(const qubit &qidx, const double theta) = 0;
  virtual void ry(const qubit &qidx, const double theta) = 0;
  virtual void rz(const qubit &qidx, const double theta) = 0;
  // U1(theta) gate
  virtual void u1(const qubit &qidx, const double theta) = 0;
  virtual void u3(const qubit &qidx, const double theta, const double phi,
                  const double lambda) = 0;
  // Reset
  virtual void reset(const qubit &qidx) = 0;
  // Measure-Z
  // Optionally retrieve the classical register bit name storing the bit result.
  // Rationale: in NISQ mode, whereby the bool value is not available when this
  // function executed, we can assign the *specific* classical reg name to hold
  // the measurement. This is required so that we can refer to this creg in
  // later NISQ-mode if statement if needed.
  virtual bool mz(const qubit &qidx,
                  std::pair<std::string, size_t> *optional_creg = nullptr) = 0;

  // Common two-qubit gates.
  virtual void cnot(const qubit &src_idx, const qubit &tgt_idx) = 0;
  virtual void cy(const qubit &src_idx, const qubit &tgt_idx) = 0;
  virtual void cz(const qubit &src_idx, const qubit &tgt_idx) = 0;
  virtual void ch(const qubit &src_idx, const qubit &tgt_idx) = 0;
  virtual void swap(const qubit &src_idx, const qubit &tgt_idx) = 0;

  // Common parameterized 2 qubit gates.
  virtual void cphase(const qubit &src_idx, const qubit &tgt_idx,
                      const double theta) = 0;
  virtual void crz(const qubit &src_idx, const qubit &tgt_idx,
                   const double theta) = 0;

  // exponential of i * theta * H, where H is an Observable pointer
  // virtual void exp(qreg q, const double theta, xacc::Observable &H) = 0;
  // virtual void exp(qreg q, const double theta, xacc::Observable *H) = 0;
  virtual void exp(qreg q, const double theta,
                   qcor::Operator& H) = 0;

  virtual void general_instruction(std::shared_ptr<xacc::Instruction> inst) = 0;

  // Submission API. Submit the constructed CompositeInstruction operating
  // on the provided AcceleratorBuffer(s) (note qreg wraps an AcceleratorBuffer)
  virtual void submit(xacc::AcceleratorBuffer *buffer) = 0;
  virtual void submit(xacc::AcceleratorBuffer **buffers,
                      const int nBuffers) = 0;

  // Some getters for the qcor runtime library.
  virtual void
  set_current_program(std::shared_ptr<qcor::CompositeInstruction> p) = 0;
  virtual std::shared_ptr<qcor::CompositeInstruction> get_current_program() = 0;
  virtual void set_current_buffer(xacc::AcceleratorBuffer *buffer) = 0;
  // Ancilla qubit allocator:
  // i.e. handle in kernel allocation.
  virtual QubitAllocator *get_anc_qubit_allocator() = 0;
};
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

extern int current_shots;
extern std::shared_ptr<QuantumRuntime> qrt_impl;
extern std::vector<std::string> kernels_in_translation_unit;
extern std::unordered_map<
    std::string, std::pair<std::vector<std::string>, std::vector<std::string>>>
    kernel_signatures_in_translation_unit;

void initialize(const std::string qpu_name, const std::string kernel_name);
void finalize();
void set_shots(int shots);
int get_shots();
void set_backend(std::string accelerator_name);
void set_backend(std::string accelerator_name, const int shots);
void set_qrt(const std::string &qrt_name);

// Common single-qubit gates.
void h(const qubit &qidx);
void x(const qubit &qidx);
void y(const qubit &qidx);
void z(const qubit &qidx);
void t(const qubit &qidx);
void tdg(const qubit &qidx);
void s(const qubit &qidx);
void sdg(const qubit &qidx);
// Reset a qubit (to zero state)
void reset(const qubit &qidx);

// broadcast across qreg
void h(qreg q);
void x(qreg q);
void y(qreg q);
void z(qreg q);
void t(qreg q);
void tdg(qreg q);
void s(qreg q);
void sdg(qreg q);
void reset(qreg qidx);

// Common single-qubit, parameterized instructions
void rx(const qubit &qidx, const double theta);
void ry(const qubit &qidx, const double theta);
void rz(const qubit &qidx, const double theta);
// U1(theta) gate
void u1(const qubit &qidx, const double theta);
void u3(const qubit &qidx, const double theta, const double phi,
        const double lambda);

// broadcast rotations across qubits
void rx(qreg qidx, const double theta);
void ry(qreg qidx, const double theta);
void rz(qreg qidx, const double theta);
// U1(theta) gate
void u1(qreg qidx, const double theta);
void u3(qreg qidx, const double theta, const double phi,
        const double lambda);

// Measure-Z and broadcast mz
bool mz(const qubit &qidx);
void mz(qreg q);

// Common two-qubit gates.
void cnot(const qubit &src_idx, const qubit &tgt_idx);
void cy(const qubit &src_idx, const qubit &tgt_idx);
void cz(const qubit &src_idx, const qubit &tgt_idx);
void ch(const qubit &src_idx, const qubit &tgt_idx);
void swap(const qubit &src_idx, const qubit &tgt_idx);

// Common parameterized 2 qubit gates.
void cphase(const qubit &src_idx, const qubit &tgt_idx, const double theta);
void crz(const qubit &src_idx, const qubit &tgt_idx, const double theta);

// Broadcast two registers
void cnot(qreg src, qreg tgt);
void cy(qreg src, qreg tgt);
void cz(qreg src, qreg tgt);
void ch(qreg src, qreg tgt);

// exponential of i * theta * H, where H is an Observable pointer
void exp(qreg q, const double theta, qcor::Operator &H);
// void exp(qreg q, const double theta, xacc::Observable *H);
// void exp(qreg q, const double theta, std::shared_ptr<xacc::Observable> H);

// Submission API. Submit the constructed CompositeInstruction operating
// on the provided AcceleratorBuffer(s) (note qreg wraps an AcceleratorBuffer)
void submit(xacc::AcceleratorBuffer *buffer);
void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers);

// Some getters for the qcor runtime library.
void set_current_program(std::shared_ptr<qcor::CompositeInstruction> p);

// Set the *runtime* buffer
void set_current_buffer(xacc::AcceleratorBuffer *buffer);
// std::shared_ptr<xacc::CompositeInstruction> getProgram();
// xacc::CompositeInstruction *program_raw_pointer();

// // Clear the current program
// void clearProgram();

// Persist bit-string result from single-bit measurements (if any)
void persistBitstring(xacc::AcceleratorBuffer *buffer);

// Get the ancilla qubit allocator:
QubitAllocator *getAncillaQubitAllocator();
} // namespace quantum

namespace xacc {

namespace internal_compiler {

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

// Placement strategy specified in the QCOR command line.
extern std::string __placement_name;
// Qubit map for DefaultPlacement transformation.
// If provided in the command line (not empty),
// we'll map qubits according to this.
extern std::vector<int> __qubit_map;
extern std::vector<int> parse_qubit_map(const char *qubit_map_str);
extern void apply_decorators(const std::string &decorator_cmdline_string);
extern std::string __qrt_env;
// Print final CompositeInstruction for backend submission
extern bool __print_final_submission;
extern std::string __print_final_submission_filename;

// Execute the pass manager on the provided kernel.
// If none provided, execute the pass manager on the current QRT kernel.
void execute_pass_manager(
    std::shared_ptr<qcor::CompositeInstruction> optional_composite = nullptr);
std::string get_native_code(std::shared_ptr<qcor::CompositeInstruction> program,
                            xacc::HeterogeneousMap options);
} // namespace internal_compiler
} // namespace xacc
namespace qcor {
// Ancilla qubit allocator:
class AncQubitAllocator : public AllocEventListener, public QubitAllocator {
public:
  static inline const std::string ANC_BUFFER_NAME = "aux_temp_buffer";
  virtual void onAllocate(xacc::internal_compiler::qubit *in_qubit) override {
    // std::cout << "Allocate: " << (void *)in_qubit << "\n";
  }

  // On deallocate: don't try to deref the qubit since it may have been gone.
  virtual void onDealloc(xacc::internal_compiler::qubit *in_qubit) override;

  virtual xacc::internal_compiler::qubit allocate() override;
  std::shared_ptr<xacc::AcceleratorBuffer> get_buffer() { return m_buffer; }

protected:
  std::vector<xacc::internal_compiler::qubit> m_qubitPool;
  // Track the list of qubit pointers for those
  // that was allocated by this Allocator.
  std::vector<xacc::internal_compiler::qubit *> m_allocatedQubits;
  std::shared_ptr<xacc::AcceleratorBuffer> m_buffer;
};
} // namespace qcor
#endif