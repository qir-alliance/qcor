#ifndef RUNTIME_QCOR_QRT_HPP_
#define RUNTIME_QCOR_QRT_HPP_

#include <memory>

namespace xacc {
class AcceleratorBuffer;
class CompositeInstruction;
class IRProvider;
}

namespace quantum {

extern std::shared_ptr<xacc::CompositeInstruction> program;
extern std::shared_ptr<xacc::IRProvider> provider;

void initialize(const std::string qpu_name, const std::string kernel_name);
void set_shots( int shots );

void h(const std::size_t qidx);
void cnot(const std::size_t src_idx, const std::size_t tgt_idx);
void mz(const std::size_t qidx);

void submit(xacc::AcceleratorBuffer * buffer);

}

#endif