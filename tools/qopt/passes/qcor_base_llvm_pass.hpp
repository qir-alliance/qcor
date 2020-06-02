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

namespace qcor {

// This class provides some base functionality for all
// qcor FunctionPasses, specifically, a mechanism for only
// operating on Functions that are annotated quantum
class QCORBaseFunctionPass : public FunctionPass {
protected:
  const char *AnnotationString = "quantum";

  QCORBaseFunctionPass(char pid) : FunctionPass(pid) {}

  std::set<Function *> annotFuncs;
  bool shouldInstrumentFunc(Function &F) {
    return annotFuncs.find(&F) != annotFuncs.end();
  }

  auto demangle(const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  }
  
public:
  virtual bool doInitialization(Module &M) override {
    getAnnotatedFunctions(&M);
    return false;
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
          }
        }
      }
    }
  }
};
} // namespace qcor
