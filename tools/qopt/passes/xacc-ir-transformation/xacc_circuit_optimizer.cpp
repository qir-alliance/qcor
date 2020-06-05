#include "qalloc"
#include "qcor_base_llvm_pass.hpp"
#include "xacc_internal_compiler.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

#include "llvm/Linker/Linker.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <regex>
#include <xacc.hpp>

using namespace llvm;
namespace {

struct FindQubitRegisters : public FunctionPass {
  static char ID;

  FindQubitRegisters() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    for (BasicBlock &b : F) {
      // Loop over instructions
      for (Instruction &i : b) {
        if (auto *call = dyn_cast<CallInst>(&i)) {
          auto f = call->getCalledFunction();

          if (f && call->getCalledFunction()->getName().str().find("qalloc") !=
                       std::string::npos) {
            auto var_name = call->getOperand(0)->getName().str();
            auto size_value = call->getOperand(1);
            if (auto *const_size = dyn_cast<ConstantInt>(size_value)) {
              llvm::errs() << "Found qreg with name " << var_name
                           << " and size " << const_size->getValue() << "\n";
              auto buffer = qalloc(const_size->getValue().getLimitedValue());
              buffer.setNameAndStore(var_name.c_str());
            } else {
              llvm::errs() << "Found qreg but do not know its size\n";
            }
          }
        }
      }
    }

    return false;
  }
};

std::map<std::string, std::string> qrt_to_xacc{
    {"h", "H"}, {"cnot", "CNOT"}, {"mz", "Measure"}};

struct CircuitOptimizer : public qcor::QCORBaseFunctionPass {
  static char ID;
  const char *AnnotationString = "quantum";

  CircuitOptimizer() : QCORBaseFunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (shouldInstrumentFunc(F) == false)
      return false;

    // llvm::errs() << "Optimizing quantum kernel with xacc.\n";

    if (!xacc::isInitialized()) {
      xacc::Initialize();
      xacc::external::load_external_language_plugins();
    }

    // Get the quantum kernel function name
    std::string kernel_name = demangle(F.getName().str().c_str());
    kernel_name = kernel_name.substr(0, kernel_name.find_first_of("("));
    // llvm::errs() << "This function name is " << kernel_name << "\n";

    // Need to construct the function prototype string
    std::string function_prototype = "void " + kernel_name + "(";
    for (auto &arg : F.args()) {
      auto arg_name = arg.getName().str();
      std::string type_str;
      llvm::raw_string_ostream rso(type_str);
      arg.getType()->print(rso);
      rso.str();
      if (type_str.find("qreg") != std::string::npos) {
        type_str = "qreg";
      }

      function_prototype += type_str + " " + arg.getName().str() + ", ";
    }
    function_prototype =
        function_prototype.substr(0, function_prototype.length() - 2) + ")";

    // Goal is to build up XACC CompositeInstruction from
    // sequential QRT calls.
    auto provider = xacc::getIRProvider("quantum");
    auto composite = provider->createComposite("tmp");
    std::vector<std::size_t> seen_qbit_idxs;
    std::vector<std::string> buffer_names;
    std::set<std::string> unique_buffer_names;

    for (BasicBlock &b : F) {

      // Loop over instructions
      for (Instruction &i : b) {

        // I have observed qreg[IDX] can be a CallInst or an InvokeInst
        // so need to check both to add IDX to seen_qbit_idxs, and qrt calls
        // are invoke insts
        if (auto *call = dyn_cast<CallInst>(&i)) {
          auto f = call->getCalledFunction();
          if (f &&
              demangle(f->getName().str().c_str()).find("qreg::operator[]") !=
                  std::string::npos) {
            if (auto *const_int = dyn_cast<ConstantInt>(call->getOperand(2))) {
              auto bit_idx = const_int->getValue().getLimitedValue();
              seen_qbit_idxs.push_back(bit_idx);
            }
            buffer_names.push_back(call->getOperand(1)->getName().str());
            unique_buffer_names.insert(call->getOperand(1)->getName().str());
          }

        } else if (auto *invoke = dyn_cast<InvokeInst>(&i)) {
          auto f = invoke->getCalledFunction();
          if (f &&
              demangle(f->getName().str().c_str()).find("qreg::operator[]") !=
                  std::string::npos) {
            if (auto *const_int =
                    dyn_cast<ConstantInt>(invoke->getOperand(2))) {
              auto bit_idx = const_int->getValue().getLimitedValue();
              seen_qbit_idxs.push_back(bit_idx);
            }
            buffer_names.push_back(invoke->getOperand(1)->getName().str());
            unique_buffer_names.insert(invoke->getOperand(1)->getName().str());
          } else if (f &&
                     demangle(f->getName().str().c_str()).find("quantum::") !=
                         std::string::npos) {

            auto qrt_call_str = demangle(f->getName().str().c_str());
            auto split = xacc::split(qrt_call_str, ':');
            auto qrt_name = split[2].substr(0, split[2].find_first_of("("));

            if (qrt_to_xacc.count(qrt_name)) {
              auto xacc_name = qrt_to_xacc[qrt_name];
              auto inst =
                  provider->createInstruction(xacc_name, seen_qbit_idxs);
              inst->setBufferNames(buffer_names);
              seen_qbit_idxs.clear();
              buffer_names.clear();
              composite->addInstruction(inst);
            } else if (qrt_name == "initialize") {
             
              // FIXME get the accelerator name...
            //   llvm::errs() << "HERE WE HAVE INIT\n";
            //   std::string type_str;
            //   llvm::raw_string_ostream rso(type_str);
            //   f->getArg(0)->getType()->print(rso);

            //   f->dump();
            //   invoke->dump();
            //   invoke->getOperand(0)->dump();
            //   if (isa<StringLiteral>(invoke->getOperand(0))) {
            //       errs() << "THIS WAS A CONST EXPR\n";
            //   }
            }
          }
        }
      }
    }

    // llvm::errs() << "XACC:\n" << composite->toString() << "\n";

    // Optimize the XACC CompositeInstruction
    auto optimizer = xacc::getIRTransformation("circuit-optimizer");
    optimizer->apply(composite, nullptr);

    // llvm::errs() << "After Opt:\n" << composite->toString() << "\n";

    // Now we want to translate this optimized CompositeInstruction
    // to a new LLVM Function instance. Use the xacc-llvm Compiler for that
    auto compiler = xacc::getCompiler("xacc-llvm");

    std::vector<std::string> as_vec;
    for (auto &b : unique_buffer_names) {
      as_vec.push_back(b);
    }

    auto opt_kernel_name = "__xacc__optimized_kernel__";
    function_prototype = std::regex_replace(
        function_prototype, std::regex(kernel_name), opt_kernel_name);

    Function *ff = &F;
    xacc::HeterogeneousMap extra_data{
        std::make_pair("accelerator", "qpp"),
        std::make_pair("kernel-name", opt_kernel_name),
        std::make_pair("buffer-names", as_vec),
        std::make_pair("function-prototype", function_prototype)};

    // Translate this CompositeInstruction to an LLVM Module IR string
    auto llvm_mod_str = compiler->translate(composite, extra_data);

    // Parse that string to a Module
    SMDiagnostic Err;
    auto mem = MemoryBuffer::getMemBuffer(llvm_mod_str)->getMemBufferRef();
    auto new_mod = llvm::parseIR(mem, Err, F.getContext());

    // Link the new module with the optimized function
    // in with the current module
    auto current_module = F.getParent();
    Linker l(*current_module);
    l.linkInModule(std::move(new_mod), Linker::Flags::OverrideFromSrc);

    // Find that optimized kernel Function
    Function *optimized_kernel;
    for (auto &F : current_module->getFunctionList()) {
      if (demangle(F.getName().str().c_str()).find(opt_kernel_name) !=
          std::string::npos) {
        errs() << "Found the optimized kernel call\n";
        optimized_kernel = &F;
      }
    }

    // Replace all uses of this Function F with
    // the new optimized kernel
    F.replaceAllUsesWith(optimized_kernel);

    // return true to indicate we modified the ir
    return true;
  }
};
} // namespace

char FindQubitRegisters::ID = 0;
char CircuitOptimizer::ID = 0;

static RegisterPass<CircuitOptimizer> XX("xacc-optimize", "", false, false);
static RegisterPass<FindQubitRegisters> X("find-qregs", "Find qubit registers",
                                          true, true);

// saving for later
// auto function_type = F.getFunctionType();
// Function *new_function = Function::Create(function_type,
// Function::ExternalLinkage,
//                                  F.getName(), F.getParent());

// llvm::errs() << src_str << "\n";

// F.getArg(0)->dump();
// //   auto qreg_f =
// //
// F.getParent()->getFunction("xacc::internal_compiler::qreg::operator[]");

// //  if (!qreg_f) llvm::errs() <<"THIS CALL WAS NULL\n" ;
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
//         llvm::errs() << "Found QRT Call " << qrt_name << "\n";
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
//                                    llvm::APInt(64, 0))};

//             llvm::errs() << "TEST: "<<
//             one_qubit_function_type->getNumParams() << "\n"; for (int j =
//             0 ; j < one_qubit_function_type->getNumParams(); j++) {
//                 one_qubit_function_type->getParamType(j)->dump();
//                 args[j]->getType()->dump();
//             }
//             auto new_inst = llvm::CallInst::Create(
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
//     llvm::errs() << "Modified BLock:\n";
//     b.dump();
//   }
// }