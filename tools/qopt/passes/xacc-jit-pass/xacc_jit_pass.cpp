#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <CompositeInstruction.hpp>
#include <qalloc.hpp>
#include <regex>
#include <xacc_internal_compiler.hpp>

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

#include "llvm_ir_visitors.hpp"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "jit_utils.hpp"

using namespace llvm;
namespace {

class FindQubitRegisters : public InstVisitor<FindQubitRegisters> {

public:
  std::map<std::string, std::size_t> buffer_to_size;

  void visitCallInst(CallInst &call) {
    auto f = call.getCalledFunction();
    if (f && f->getName().str().find("qalloc") != std::string::npos) {
      auto var_name = call.getOperand(0)->getName().str();
      auto size_value = call.getOperand(1);
      if (auto *const_size = dyn_cast<ConstantInt>(size_value)) {
        llvm::errs() << "Found qreg with name " << var_name << " and size "
                     << const_size->getValue() << "\n";
        auto size = const_size->getValue().getLimitedValue();
        buffer_to_size.insert({var_name, size});
      } else {
        llvm::errs() << "Found qreg but do not know its size\n";
      }
    }
  }
};

struct JITCircuitOptimizer : public ModulePass {
  static char ID;
  const char *AnnotationString = "quantum";

  std::map<std::string, std::string> func_name_to_wrapper;
  std::map<std::string, std::string> real_to_mangled_names;

  JITCircuitOptimizer() : ModulePass(ID) {}

  std::set<Function *> annotFuncs;
  bool isQuantumKernel(Function &F) {
    return annotFuncs.find(&F) != annotFuncs.end();
  }

  auto demangle(const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  }

  void packFunctionArguments(Module *module) {
    auto &ctx = module->getContext();
    llvm::IRBuilder<> builder(ctx);
    DenseSet<llvm::Function *> interfaceFunctions;
    for (auto &func : module->getFunctionList()) {
      if (func.isDeclaration()) {
        continue;
      }
      if (interfaceFunctions.count(&func)) {
        continue;
      }
      if (!isQuantumKernel(func)) {
        continue;
      }

      auto ciType = module->getTypeByName("class.xacc::CompositeInstruction");
      errs() << "Dumpy compoisite\n";
      ciType->dump();

      auto gv_exec =
          module->getNamedGlobal("_ZN4xacc17internal_compiler9__executeE");

      // Given a function `foo(<...>)`, define the interface function
      // `mlir_foo(i8**)`.
      auto newType = llvm::FunctionType::get(
          ciType->getPointerTo(), builder.getInt8PtrTy()->getPointerTo(),
          /*isVarArg=*/false);
      auto newName = "__qcor__" + func.getName().str();
      auto funcCst = module->getOrInsertFunction(newName, newType);
      llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());
      interfaceFunctions.insert(interfaceFunc);

      func_name_to_wrapper.insert(
          {func.getName().str(), interfaceFunc->getName().str()});

      // Extract the arguments from the type-erased argument list and cast
      // them to the proper types.
      auto bb = llvm::BasicBlock::Create(ctx);
      bb->insertInto(interfaceFunc);
      builder.SetInsertPoint(bb);
      llvm::Value *argList = interfaceFunc->arg_begin();

      SmallVector<llvm::Value *, 8> args;
      args.reserve(llvm::size(func.args()));
      for (auto &indexedArg : llvm::enumerate(func.args())) {
        errs() << "looping wrapper ";
        errs() << indexedArg.index() << "\n";
        indexedArg.value().dump();
        llvm::Value *argIndex = llvm::Constant::getIntegerValue(
            builder.getInt64Ty(), APInt(64, indexedArg.index()));
        errs() << "argidx:\n";
        argIndex->dump();
        llvm::Value *argPtrPtr = builder.CreateGEP(argList, argIndex);
        errs() << "arg gep ptr ptr:\n";
        argPtrPtr->dump();

        llvm::Value *argPtr = builder.CreateLoad(argPtrPtr);
        errs() << "arg ptr load:\n";
        argPtr->dump();

        argPtr = builder.CreateBitCast(
            argPtr, indexedArg.value().getType()->getPointerTo());
        errs() << "arg ptr bitcast:\n";
        argPtr->dump();
        llvm::Value *arg = builder.CreateLoad(argPtr);
        errs() << "arg load:\n";
        arg->dump();
        args.push_back(arg);
      }

      auto turn_off_exec = builder.getInt8(0);
      builder.CreateStore(turn_off_exec, gv_exec);

      // Call the implementation function with the extracted arguments.
      llvm::Value *tmp_result = builder.CreateCall(&func, args);

      auto turn_on_exec = builder.getInt8(1);
      builder.CreateStore(turn_on_exec, gv_exec);

      FunctionType *get_prog_type =
          FunctionType::get(ciType->getPointerTo(), {}, false);

      auto get_prog_func = module->getOrInsertFunction(
          "_ZN7quantum19program_raw_pointerEv", get_prog_type);

      auto *result = builder.CreateCall(get_prog_func);

      builder.CreateRet(result);

      interfaceFunc->dump();
      // Assuming the result is one value, potentially of type `void`.
      //   if (!result->getType()->isVoidTy()) {
      //   llvm::Value *retIndex = llvm::Constant::getIntegerValue(
      //       builder.getInt64Ty(), APInt(64, llvm::size(func.args())));
      //   llvm::Value *retPtrPtr = builder.CreateGEP(argList, retIndex);
      //   llvm::Value *retPtr = builder.CreateLoad(retPtrPtr);
      //   retPtr = builder.CreateBitCast(retPtr,
      //   result->getType()->getPointerTo()); builder.CreateStore(result,
      //   retPtr);
      //   //   }

      //   // Turn on execution
      //   //   FunctionType *execute_on =
      //   //       FunctionType::get(builder.getVoidTy(), {}, false);
      //   //   auto execute_on_func = module->getOrInsertFunction(
      //   //       "_ZN4xacc17internal_compiler14__execution_onEv",
      //   execute_on);
      //   //   builder.CreateCall(execute_on_func);

      //   // The interface function returns void.
      //   builder.CreateRetVoid();
    }
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
            errs() << "inserting: " << FUNC->getName().str() << "\n";
            annotFuncs.insert(FUNC);
          }
        }
      }
    }
  }

  bool runOnModule(Module &M) override {

    for (auto &F : M) {
      real_to_mangled_names.insert(
          {demangle(F.getName().str().c_str()), F.getName().str()});
    }

    FindQubitRegisters visitor;
    visitor.visit(M);

    for (auto &[k, v] : visitor.buffer_to_size) {
      errs() << "Buffer seen: " << k << ", " << v << "\n";
    }

    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(
          MangledNameStream,
          "jit_test(xacc::internal_compiler::qreg, std::vector<double>)",
          M.getDataLayout());
    }
    errs() << "MANGLED: " << MangledName << "\n";

    packFunctionArguments(&M);
    // M.dump();
    std::set<CallInst *> seen_quantum_kernel_calls;
    for (Function *F : annotFuncs) {
      errs() << " WORKING ON \n";
      errs() << F->getName().str() << "\n";

      errs() << "Optimizing quantum kernel with xacc.\n";

      // Get the quantum kernel function name
      std::string kernel_name = demangle(F->getName().str().c_str());
      kernel_name = kernel_name.substr(0, kernel_name.find_first_of("("));
      errs() << "This function name is " << kernel_name << "\n";

      // We should only run this on entry-level quantum kernels
      // i.e. only those whose parent is not a quantum kernel

      if (!xacc::isInitialized()) {
        xacc::Initialize();
        xacc::external::load_external_language_plugins();
      }

      auto jit = cantFail(qcor::XACCJIT::Create());

      auto mod_ptr = llvm::CloneModule(M);

      auto error = jit->addModule(std::move(mod_ptr));
      if (error) {
        errs() << "adding mod error\n";
      } else {
        errs() << error << ", error adding mod\n";
      }

      xacc::internal_compiler::qreg q(2);
      void ** args = new void * [1];
      args[0] = &q;

    //   errs() << "hello: " << q.size() << ", " << q.results()->name() <<" \n";
    //   std::vector<void *> args;
    //   args.push_back(&q);

      auto symbol = cantFail(jit->lookup(func_name_to_wrapper[F->getName()]));
      auto rawFPtr = symbol.getAddress();
      auto fptr =
          reinterpret_cast<xacc::CompositeInstruction *(*)(void **)>(rawFPtr);

      auto p = (*fptr)(args);

      // return true to indicate we modified the ir
      return false;
    }

    return false;
  }
}; // namespace
} // namespace

char JITCircuitOptimizer::ID = 0;

static RegisterPass<JITCircuitOptimizer> XX("xacc-jit-optimize", "", false,
                                            false);

static void registerMyPass(const PassManagerBuilder &, PassManagerBase &PM) {
  PM.add(new JITCircuitOptimizer());
}
static RegisterStandardPasses
    RegisterMyPass(PassManagerBuilder::EP_EarlyAsPossible, registerMyPass);