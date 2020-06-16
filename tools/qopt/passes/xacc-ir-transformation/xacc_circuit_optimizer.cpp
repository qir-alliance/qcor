
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#include <regex>

#include "llvm_ir_visitors.hpp"

using namespace llvm;
namespace {

struct CircuitOptimizer : public ModulePass {
  static char ID;
  const char *AnnotationString = "quantum";

  CircuitOptimizer() : ModulePass(ID) {}

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

  bool runOnModule(Module &M) override {

    std::set<CallInst *> seen_quantum_kernel_calls;
    for (Module::iterator f_iter = M.begin(), E = M.end(); f_iter != E;
         ++f_iter) {
      Function &F = *f_iter;
      // If this F is quantum and it is an entry point,
      // i.e. only those quantum kernels whose parent is not a
      // another quantum kernel
      if (!isQuantumKernel(F))
        continue;

      errs() << "Optimizing quantum kernel with xacc.\n";

      // Get the quantum kernel function name
      std::string kernel_name = demangle(F.getName().str().c_str());
      kernel_name = kernel_name.substr(0, kernel_name.find_first_of("("));
      errs() << "This function name is " << kernel_name << "\n";

      // We should only run this on entry-level quantum kernels
      // i.e. only those whose parent is not a quantum kernel

      // Search for the CallInst representing this Quantum Kernel
      qcor::SearchForCallsToThisFunction search(&F);
      search.visit(M);
      auto call_for_this_F = search.found_call;
      // If found, is the parent function a quantum kernel?
      // if it is, then we don't want to optimize this.
      if (call_for_this_F &&
          isQuantumKernel(*call_for_this_F->getParent()->getParent())) {
        seen_quantum_kernel_calls.insert(call_for_this_F);
        continue;
      }

      if (!xacc::isInitialized()) {
        xacc::Initialize();
        xacc::external::load_external_language_plugins();
      }

      // Get all function variable names
      std::vector<std::string> var_names;
      for (auto &arg : F.args()) {
        auto arg_name = arg.getName().str();
        var_names.push_back(arg_name);
      }

      qcor::FindFunctionVariableStoreInsts find_stores(var_names);
      find_stores.visit(F);

      // Start off by inlining all quantum kernels that are
      // children of this entry-level quantum kernel
      for (auto &call_to_be_inlined : seen_quantum_kernel_calls) {
        InlineFunctionInfo ifi;
        auto success = InlineFunction(call_to_be_inlined, ifi);
        errs() << "was able to be inlined? " << success.message << "\n";
      }

      // One thing to look out for is after inlining, qreg(qreg&) might have
      // been called, if so, then the qreg buffer name will
      // show up as different from original. So here I want to
      // loop through, find these calls and build up map of new_names to
      // old_qreg_names
      // call void
      // @_ZN4xacc17internal_compiler4qregC1ERKS1_(%"class.xacc::internal_compiler::qreg"*
      // %agg.tmp45, %"class.xacc::internal_compiler::qreg"* dereferenceable(8)
      // %q)
      std::map<std::string, std::string> new_to_old_qreg_names;
      for (BasicBlock &b : F) {

        // Loop over instructions
        for (Instruction &i : b) {

          if (auto *call = dyn_cast<CallInst>(&i)) {
            auto f = call->getCalledFunction();
            if (f && demangle(f->getName().str().c_str())
                             .find("xacc::internal_compiler::qreg::qreg(xacc::"
                                   "internal_compiler::qreg const&)") !=
                         std::string::npos) {
              errs() << "We have found qreg copy consturctor\n";
              call->dump();

              errs() << call->getOperand(0)->getName().str() << ", "
                     << call->getOperand(1)->getName().str() << "\n";
              new_to_old_qreg_names.insert(
                  {call->getOperand(0)->getName().str(),
                   call->getOperand(1)->getName().str()});
            }
          }
        }
      }

      // NOTE now we assume that we have flattened qrt calls...
      F.dump();
      // Map LLVM IR made up of flatted qrt calls to
      // an XACC CompositeInstruction instance
      qcor::LLVM_IR_To_XACC to_xacc_visitor(new_to_old_qreg_names);
      to_xacc_visitor.visit(F);
      auto composite = to_xacc_visitor.composite;

      errs() << "XACC:\n" << composite->toString() << "\n";

      // Optimize the XACC CompositeInstruction
      auto optimizer = xacc::getIRTransformation("circuit-optimizer");
      optimizer->apply(composite, nullptr);

      // Map the Optimized CompositeInstruction back to LLVM IR qrt calls
      qcor::XACC_To_LLVM_IR visitor(F.getParent(), find_stores.stores,
                                    composite.get());
      visitor.visit(F);

      F.dump();

      // Visitor stores the basic block the first Instruction is found in
      // we want to set the normal destination of that block to the
      // execution block.
      auto &last_inst_iter = visitor.basic_block->back();
      auto old_bb = dyn_cast<InvokeInst>(&last_inst_iter)->getNormalDest();
      if (old_bb != visitor.execution_block) {

        dyn_cast<InvokeInst>(&last_inst_iter)
            ->setNormalDest(visitor.execution_block);

        // Erase the old normal destination
        old_bb->eraseFromParent();
      }

      // Loop over the basic blocks and clean up all
      // those that aren't being used, don't have predecessors
      while (true) {
        std::vector<BasicBlock *> remove_these;
        for (BasicBlock &b : F) {
          auto bname = b.getName().str();
          if (!b.hasNPredecessorsOrMore(1) &&
              (bname.find("invoke.cont") != std::string::npos ||
               bname.find("lpad") != std::string::npos ||
               bname.find("if.then") != std::string::npos ||
               bname.find(".exit") != std::string::npos)) {

            remove_these.push_back(&b);

          } else if (b.getInstList().size() == 0) {
            remove_these.push_back(&b);
          }
        }
        if (remove_these.empty()) {
          break;
        }

        for (auto b : remove_these) {
          b->eraseFromParent();
        }
      }

      F.dump();

      // return true to indicate we modified the ir
      return true;
    }
    return false;
  }
}; // namespace
} // namespace

char CircuitOptimizer::ID = 0;

static RegisterPass<CircuitOptimizer> XX("xacc-optimize", "", false, false);

// saving for later

// errs() << "After Opt:\n" << composite->toString() << "\n";

// Now we want to translate this optimized CompositeInstruction
// to a new LLVM Function instance. Use the xacc-llvm Compiler for that
// auto compiler = xacc::getCompiler("xacc-llvm");

// std::vector<std::string> as_vec;
// for (auto &b : unique_buffer_names) {
//   as_vec.push_back(b);
// }

// auto opt_kernel_name = "__xacc__optimized_kernel__";
// function_prototype = std::regex_replace(
//     function_prototype, std::regex(kernel_name), opt_kernel_name);

// Function *ff = &F;
// xacc::HeterogeneousMap extra_data{
//     std::make_pair("accelerator", "qpp"),
//     std::make_pair("kernel-name", opt_kernel_name),
//     std::make_pair("buffer-names", as_vec),
//     std::make_pair("function-prototype", function_prototype)};

// // Translate this CompositeInstruction to an LLVM Module IR string
// auto llvm_mod_str = compiler->translate(composite, extra_data);

// // Parse that string to a Module
// SMDiagnostic Err;
// auto mem = MemoryBuffer::getMemBuffer(llvm_mod_str)->getMemBufferRef();
// auto new_mod = parseIR(mem, Err, F.getContext());

// // Link the new module with the optimized function
// // in with the current module
// auto current_module = F.getParent();
// Linker l(*current_module);
// l.linkInModule(std::move(new_mod), Linker::Flags::OverrideFromSrc);

// // Find that optimized kernel Function
// Function *optimized_kernel;
// for (auto &F : current_module->getFunctionList()) {
//   if (demangle(F.getName().str().c_str()).find(opt_kernel_name) !=
//       std::string::npos) {
//     errs() << "Found the optimized kernel call\n";
//     optimized_kernel = &F;
//   }
// }

// // Replace all uses of this Function F with
// // the new optimized kernel
// F.replaceAllUsesWith(optimized_kernel);

// auto function_type = F.getFunctionType();
// Function *new_function = Function::Create(function_type,
// Function::ExternalLinkage,
//                                  F.getName(), F.getParent());

// errs() << src_str << "\n";

// F.getArg(0)->dump();
// //   auto qreg_f =
// //
// F.getParent()->getFunction("xacc::internal_compiler::qreg::operator[]");

// //  if (!qreg_f) errs() <<"THIS CALL WAS NULL\n" ;
// auto &ctx = F.getContext();
// //   FunctionType *printf_type = TypeBuilder<int(char *, ...),
// //   false>::get(ctx);
// auto mod = F.getParent();

// // Goal is now to walk the tree again, search for first qrt quantum call
// // once found, replace it with new call insts for each Instruction in
// // composite and then delete all instructions up until the if stmt
// bool found_first_qinst = false;
// for (BasicBlock &b : F) {

//   // Loop over instructions
//   for (Instruction &i : b) {
//     // will want this
//     //   Instruction& inst = *I;
//     //  I = inst.eraseFromParent();
//     // if (!Inst->use_empty())
//     // Inst->replaceAllUsesWith(UndefValue::get(Inst->getType()));

//     if (auto *invoke = dyn_cast<InvokeInst>(&i)) {
//       auto f = invoke->getCalledFunction();
//       if (f && demangle(f->getName().str().c_str()).find("quantum::") !=
//                    std::string::npos) {
//         auto qrt_call_str = demangle(f->getName().str().c_str());
//         auto split = xacc::split(qrt_call_str, ':');
//         auto qrt_name = split[2].substr(0, split[2].find_first_of("("));
//         errs() << "Found QRT Call " << qrt_name << "\n";
//         if (qrt_to_xacc.count(qrt_name)) {

//           // This is a quantum qrt call.
//           if (!found_first_qinst) {
//             found_first_qinst = true;

//             // for all xacc::Instructions, create
//             // LLVM IR Instructions. This means we need
//             // nRequiredBits() qreg[idx] calls followed by
//             // the quantum:: qrt call. Looks like this for
//             // quantum::h(q[0]);
//             // call void
//             //
//             @_ZN4xacc17internal_compiler4qregixB5cxx11Em(%"struct.std::pair"*
//             // sret %ref.tmp9,
//             // %"class.xacc::internal_compiler::qreg"* %q, i64 0)
//             // invoke void
//             //
//             @_ZN7quantum1hERKSt4pairINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEmE(
//             // %"struct.std::pair"* dereferenceable(40) %ref.tmp9)

//             auto xacc_inst = composite->getInstruction(0);
//             auto n_bits = xacc_inst->nRequiredBits();

//             // Here we want to call qreg[0] or something like that
//             // (1) so we need to allocate a temp struct.std::pair
//             // (2) create the operator[] call inst
//             // add that to
//             Function *func = dyn_cast<Function>(
//                 mod->getOrInsertFunction(
//                        "xacc::internal_compiler::qreg::operator[]",
//                        one_qubit_function_type)
//                     .getCallee());

//             //   %ref.tmp9 = alloca %"struct.std::pair", align 8
//             AllocaInst *pair_ret =
//                 new AllocaInst(std_pair_type_qubit, 0, "tmp_qubit",
//                 invoke);
//             auto qreg_ret = ReturnInst::Create(ctx, pair_ret);
//             ArrayRef<Value *> args{qreg_ret,
//             dyn_cast<Value>(F.getArg(0)),
//                                    ConstantInt::get(IntegerType::getInt64Ty(ctx),
//                                    APInt(64, 0))};

//             errs() << "TEST: "<<
//             one_qubit_function_type->getNumParams() << "\n"; for (int j =
//             0 ; j < one_qubit_function_type->getNumParams(); j++) {
//                 one_qubit_function_type->getParamType(j)->dump();
//                 args[j]->getType()->dump();
//             }
//             auto new_inst = CallInst::Create(
//                 one_qubit_function_type, func, args,
//                 "xacc::internal_compiler::qreg::operator[]", invoke);

//             // Type::
//             //   b.getInstList().insert(invoke, new_inst); // insert
//             //   new_inst before invoke, invoke->eraseFromParent();

//           } else {
//             // We've found the first one, now we're seeing others,
//             // but we'll just delete them plus the nRequiredBits()
//             // preceeding calls.
//           }
//         }
//       }
//     }
//   }
//   if (found_first_qinst) {
//     errs() << "Modified BLock:\n";
//     b.dump();
//   }
// }