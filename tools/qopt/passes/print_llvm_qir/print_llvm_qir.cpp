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

    // Leaving here for now, print callinsts example...

    for (BasicBlock &b : F) {
      // for (Function::iterator bb = F.begin(), e = F.end(); bb != e; ++bb) {
      //   BasicBlock &b = *bb;
      if (b.getName().str().find("for.cond") != std::string::npos) {
        llvm::errs() << "BasicBlock Name: " << b.getName().str() << "\n";

        // Loop over instructions
        for (Instruction &i : b) {

          llvm::errs() << "GET OPCODENAME: " << i.getOpcodeName() << "\n";
          if (isa<llvm::ICmpInst>(i)) {
            llvm::errs() << "we have a compare inst\n";
            auto *cmp = dyn_cast<ICmpInst>(&i);
            cmp->dump();
            llvm::errs() << "HELLO: " << cmp->getPredicate() << ", "
                         << cmp->getNumOperands() << "\n";
            Value *LHS = cmp->getOperand(0);
            Value *RHS = cmp->getOperand(1);
            if (isa<ConstantInt>(RHS)) {
              auto c = dyn_cast<ConstantInt>(RHS);
              auto val = c->getValue();
              llvm::errs() << "This is an int type " << val << "\n";
            }
          } else if (isa<LoadInst>(&i)) {
            llvm::errs() << "we have a load inst " << i.getNumOperands() << "\n";
            auto op = i.getOperand(0);
            llvm::errs() << "loop var name name = " << op->getName().str() << "\n";
            op->dump();
            if (isa<AllocaInst>(op)) {
            //   auto c = dyn_cast<ConstantInt>(RHS);
            //   auto val = c->getValue();
              auto const_int = dyn_cast<ConstantInt>(dyn_cast<AllocaInst>(op)->getOperand(0));
              llvm::errs() << "This is an alloc type for load inst \n";
              llvm::errs() << const_int->getValue() << "\n";
            }
          }
        }
      }
    }

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