#ifndef QCOR_XACC_LLVM_UTIL_HPP_
#define QCOR_XACC_LLVM_UTIL_HPP_

#include "xacc.hpp"

using namespace xacc;

namespace qcor {

// The goal of this class is to provide a compile
// implementation that maps LLVM IR string to a
// CompositeInstruction, and to implement translate
// to take a CompositeInstruction and create LLVM IR
// (using the usual qrt calls).
class LLVMCompiler : public xacc::Compiler {
public:
  std::shared_ptr<xacc::IR> compile(const std::string &src,
                                    std::shared_ptr<Accelerator> acc) override;

  std::shared_ptr<xacc::IR> compile(const std::string &src) override {
    return compile(src, nullptr);
  }

  const std::string
  translate(std::shared_ptr<CompositeInstruction> program) override {
    HeterogeneousMap empty;
    return translate(program, empty);
  }

  const std::string
  translate(std::shared_ptr<CompositeInstruction> program,
            HeterogeneousMap &options) override;

  const std::string name() const override { return "xacc-llvm"; }

  const std::string description() const override { return ""; }

  virtual ~LLVMCompiler() {}
};

} // namespace qcor

#endif