#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <cxxabi.h>
#include <iostream>
#include <set>

using namespace llvm;

namespace {

struct PrintKernelQIR : public FunctionPass {
  static char ID;
  const char *AnnotationString = "quantum";

  PrintKernelQIR() : FunctionPass(ID) {}

  std::set<Function *> annotFuncs;

  virtual bool doInitialization(Module &M) override {
    getAnnotatedFunctions(&M);
    return false;
  }

  bool shouldInstrumentFunc(Function &F) {
    return annotFuncs.find(&F) != annotFuncs.end();
  }

  void getAnnotatedFunctions(Module *M) {
    for (Module::global_iterator I = M->global_begin(), E = M->global_end();
         I != E; ++I) {

      if (I->getName() == "llvm.global.annotations") {
        ConstantArray *CA = dyn_cast<ConstantArray>(I->getOperand(0));
        for (auto OI = CA->op_begin(); OI != CA->op_end(); ++OI) {
          ConstantStruct *CS = dyn_cast<ConstantStruct>(OI->get());
          Function *FUNC = dyn_cast<Function>(CS->getOperand(0)->getOperand(0));
          GlobalVariable *AnnotationGL =
              dyn_cast<GlobalVariable>(CS->getOperand(1)->getOperand(0));
          StringRef annotation =
              dyn_cast<ConstantDataArray>(AnnotationGL->getInitializer())
                  ->getAsCString();
          if (annotation.compare(AnnotationString) == 0) {
            annotFuncs.insert(FUNC);
            // errs() << "Found annotated function " << FUNC->getName()<<"\n";
          }
        }
      }
    }
  }

  bool runOnFunction(Function &F) override {
    if (shouldInstrumentFunc(F) == false)
      return false;
    errs() << "Instrumenting " << F.getName() << "\n";
    F.dump();

    auto demangle = [](const char *name) {
      int status = -1;

      std::unique_ptr<char, void (*)(void *)> res{
          abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
      return (status == 0) ? res.get() : std::string(name);
    };
    for (Function::iterator bb = F.begin(), e = F.end(); bb != e; ++bb) {
      for (BasicBlock::iterator i = bb->begin(), e = bb->end(); i != e; ++i) {
        Instruction &ii = *i;

        if (isa<CallInst>(ii)) {
          Function *fun = cast<CallInst>(ii).getCalledFunction();
          std::cout << "GET OPCODENAME: " << i->getOpcodeName() << ", "
                    << demangle(fun->getName().str().c_str()) << "\n";
        }
      }
    }

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
    RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible,
                   registerPrintIRPass);
