#include "qcor_base_llvm_pass.hpp"

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
    
    // Leaving here for now, print callinsts example...

    // for (Function::iterator bb = F.begin(), e = F.end(); bb != e; ++bb) {
    //   for (BasicBlock::iterator i = bb->begin(), e = bb->end(); i != e; ++i)
    //   {
    //     Instruction &ii = *i;

    //     if (isa<CallInst>(ii)) {
    //       Function *fun = cast<CallInst>(ii).getCalledFunction();
    //       std::cout << "GET OPCODENAME: " << i->getOpcodeName() << ", "
    //                 << demangle(fun->getName().str().c_str()) << "\n";
    //     }
    //   }
    // }

    return false;
  }
};
} // namespace

char PrintKernelQIR::ID = 0;

static void registerPrintIRPass(const PassManagerBuilder &,
                                legacy::PassManagerBase &PM) {
  PM.add(new PrintKernelQIR());
}
static RegisterStandardPasses
    RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible, registerPrintIRPass);
