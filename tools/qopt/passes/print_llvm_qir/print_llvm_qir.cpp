#include "qcor_base_llvm_pass.hpp"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

namespace {

struct PrintKernelQIR : public qcor::QCORBaseFunctionPass {
  static char ID;
  const char *AnnotationString = "quantum";

  PrintKernelQIR() : QCORBaseFunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (shouldInstrumentFunc(F) == false)
      return false;

    // This pass just dumps all quantum kernel IR
    F.dump();

    return false;
  }
};
} // namespace

char PrintKernelQIR::ID = 0;

static RegisterPass<PrintKernelQIR>
    X("print-qir", "Print the LLVM IR associated with Quantum Kernels", false,
      false);

static RegisterStandardPasses Y(PassManagerBuilder::EP_EarlyAsPossible,
                                [](const PassManagerBuilder &Builder,
                                   legacy::PassManagerBase &PM) {
                                  PM.add(new PrintKernelQIR());
                                });