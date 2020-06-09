#ifndef RUNTIME_QCOR_QRT_HPP_
#define RUNTIME_QCOR_QRT_HPP_

#include "qalloc.hpp"
#include <CompositeInstruction.hpp>
#include <memory>

using namespace xacc::internal_compiler;

namespace xacc {
class AcceleratorBuffer;
class CompositeInstruction;
class IRProvider;
class Observable;

namespace internal_compiler {
// Current runtime controlled bit indices
// (if the current kernel is wrapped in a Controlled block)
extern std::vector<int> __controlledIdx;

void simplified_qrt_call_one_qbit(const char * gate_name,
                                 const char * buffer_name,
                                 const std::size_t idx);
void simplified_qrt_call_one_qbit_one_param(const char * gate_name,
                                           const char * buffer_name,
                                           const std::size_t idx,
                                           const double parameter);
void simplified_qrt_call_two_qbits(const char * gate_name,
                                  const char * buffer_name_1,
                                  const char * buffer_name_2,
                                  const std::size_t src_idx,
                                  const std::size_t tgt_idx);

} // namespace internal_compiler
} // namespace xacc

namespace quantum {

extern std::shared_ptr<xacc::CompositeInstruction> program;
extern std::shared_ptr<xacc::IRProvider> provider;

void initialize(const std::string qpu_name, const std::string kernel_name);
void set_shots(int shots);
void one_qubit_inst(const std::string &name, const qubit &qidx,
                    std::vector<double> parameters = {});
void two_qubit_inst(const std::string &name, const qubit &qidx1,
                    const qubit &qidx2, std::vector<double> parameters = {});

void h(const qubit &qidx);
void x(const qubit &qidx);
void t(const qubit &qidx);
void tdg(const qubit &qidx);

void rx(const qubit &qidx, const double theta);
void ry(const qubit &qidx, const double theta);
void rz(const qubit &qidx, const double theta);
// U1(theta) gate
void u1(const qubit &qidx, const double theta);

void mz(const qubit &qidx);

// Two-qubit gates:
void cnot(const qubit &src_idx, const qubit &tgt_idx);
void cy(const qubit &src_idx, const qubit &tgt_idx);
void cz(const qubit &src_idx, const qubit &tgt_idx);
void ch(const qubit &src_idx, const qubit &tgt_idx);
void swap(const qubit &src_idx, const qubit &tgt_idx);
void cphase(const qubit &src_idx, const qubit &tgt_idx, const double theta);
void crz(const qubit &src_idx, const qubit &tgt_idx, const double theta);

void exp(qreg q, const double theta, xacc::Observable *H);
void exp(qreg q, const double theta, std::shared_ptr<xacc::Observable> H);

void submit(xacc::AcceleratorBuffer *buffer);
void submit(xacc::AcceleratorBuffer **buffers, const int nBuffers);

std::shared_ptr<xacc::CompositeInstruction> getProgram();
void clearProgram();

} // namespace quantum

#endif