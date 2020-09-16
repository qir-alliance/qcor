#ifndef RUNTIME_QCOR_QRT_HPP_
#define RUNTIME_QCOR_QRT_HPP_

#include "qalloc.hpp"
#include "CompositeInstruction.hpp"
#include "Identifiable.hpp"
#include <memory>

using namespace xacc::internal_compiler;

namespace xacc {
class AcceleratorBuffer;
class CompositeInstruction;
class IRProvider;
class Observable;
} // namespace xacc

namespace quantum {

class QuantumRuntime : public xacc::Identifiable {

public:

virtual void initialize(const std::string kernel_name) = 0;

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

// Measure-Z
virtual bool mz(const qubit &qidx) = 0;

// Common two-qubit gates.
virtual void cnot(const qubit &src_idx, const qubit &tgt_idx) = 0;
virtual void cy(const qubit &src_idx, const qubit &tgt_idx) = 0;
virtual void cz(const qubit &src_idx, const qubit &tgt_idx) = 0;
virtual void ch(const qubit &src_idx, const qubit &tgt_idx) = 0;
virtual void swap(const qubit &src_idx, const qubit &tgt_idx) = 0;

// Common parameterized 2 qubit gates.
virtual void cphase(const qubit &src_idx, const qubit &tgt_idx, const double theta) = 0;
virtual void crz(const qubit &src_idx, const qubit &tgt_idx, const double theta) = 0;

// exponential of i * theta * H, where H is an Observable pointer
virtual void exp(qreg q, const double theta, xacc::Observable &H) = 0;
virtual void exp(qreg q, const double theta, xacc::Observable *H) = 0;
virtual void exp(qreg q, const double theta, std::shared_ptr<xacc::Observable> H) = 0;

// Submission API. Submit the constructed CompositeInstruction operating
// on the provided AcceleratorBuffer(s) (note qreg wraps an AcceleratorBuffer)
virtual void submit(xacc::AcceleratorBuffer *buffer) = 0;
virtual void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers) = 0;

// Some getters for the qcor runtime library. 
virtual void set_current_program(std::shared_ptr<xacc::CompositeInstruction> p) = 0;
virtual std::shared_ptr<xacc::CompositeInstruction> get_current_program() = 0;
virtual void set_current_buffer(xacc::AcceleratorBuffer* buffer) = 0;
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

void initialize(const std::string qpu_name, const std::string kernel_name);
void set_shots(int shots);
int get_shots();
void set_backend(std::string accelerator_name);
void set_backend(std::string accelerator_name, const int shots);

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
bool mz(const qubit &qidx);

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

// Set the *runtime* buffer
void set_current_buffer(xacc::AcceleratorBuffer* buffer);
// std::shared_ptr<xacc::CompositeInstruction> getProgram();
// xacc::CompositeInstruction *program_raw_pointer();

// // Clear the current program
// void clearProgram();

// Persist bit-string result from single-bit measurements (if any)
void persistBitstring(xacc::AcceleratorBuffer *buffer);
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
extern void apply_decorators(const std::string& decorator_cmdline_string);
extern std::string __qrt_env;
void execute_pass_manager();

} // namespace internal_compiler
} // namespace xacc

#endif