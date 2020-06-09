#pragma once
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

#include "xacc.hpp"
#include <Instruction.hpp>
#include <Utils.hpp>
#include <cxxabi.h>

using namespace llvm;

namespace qcor {

std::map<std::string, std::string> qrt_to_xacc{
    {"h", "H"},           {"rz", "Rz"},     {"ry", "Ry"},   {"rx", "Rx"},
    {"x", "X"},           {"y", "Y"},       {"z", "Z"},     {"s", "S"},
    {"t", "T"},           {"sdg", "Sdg"},   {"tdg", "Tdg"}, {"cy", "CY"},
    {"cz", "CZ"},         {"swap", "Swap"}, {"crz", "CRZ"}, {"ch", "CH"},
    {"cphase", "CPhase"}, {"i", "I"},       {"u", "U"},     {"u1", "U1"},
    {"cnot", "CNOT"},     {"mz", "Measure"}};

class FindFunctionVariableStoreInsts
    : public InstVisitor<FindFunctionVariableStoreInsts> {
protected:
  std::vector<std::string> function_variable_names;

public:
  std::map<std::string, StoreInst *> stores;

  FindFunctionVariableStoreInsts(std::vector<std::string> &func_var_names)
      : function_variable_names(func_var_names) {}

  void visitStoreInst(StoreInst &store) {
    auto name = store.getOperand(1)->getName().str();
    auto tmp_name = xacc::split(name, '.')[0];
    if (xacc::container::contains(function_variable_names, tmp_name)) {
      stores.insert({name, &store});
    }
  }
};

class LLVM_IR_To_XACC : public InstVisitor<LLVM_IR_To_XACC> {
protected:
  std::shared_ptr<xacc::IRProvider> provider;
  std::vector<std::size_t> seen_qbit_idxs;
  std::vector<std::string> buffer_names;
  std::set<std::string> unique_buffer_names;
  std::map<std::string, std::string> new_to_old_qreg_names;

public:
  std::shared_ptr<xacc::CompositeInstruction> composite;

  LLVM_IR_To_XACC(std::map<std::string, std::string> &m)
      : new_to_old_qreg_names(m) {
    provider = xacc::getIRProvider("quantum");
    composite = provider->createComposite("tmp");
  }

  auto demangle(const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  }

  void visitCallInst(CallInst &call) {
    auto f = call.getCalledFunction();
    if (f && demangle(f->getName().str().c_str()).find("qreg::operator[]") !=
                 std::string::npos) {
      if (auto *const_int = dyn_cast<ConstantInt>(call.getOperand(2))) {
        auto bit_idx = const_int->getValue().getLimitedValue();
        seen_qbit_idxs.push_back(bit_idx);
      }
      auto seen_buf_name = call.getOperand(1)->getName().str();
      if (new_to_old_qreg_names.count(seen_buf_name)) {
        seen_buf_name = new_to_old_qreg_names[seen_buf_name];
      }
      buffer_names.push_back(seen_buf_name);
      unique_buffer_names.insert(seen_buf_name);
    }
  }

  void visitInvokeInst(InvokeInst &invoke) {
    auto f = invoke.getCalledFunction();
    if (f && demangle(f->getName().str().c_str()).find("qreg::operator[]") !=
                 std::string::npos) {
      if (auto *const_int = dyn_cast<ConstantInt>(invoke.getOperand(2))) {
        auto bit_idx = const_int->getValue().getLimitedValue();
        seen_qbit_idxs.push_back(bit_idx);
      }

      auto seen_buf_name = invoke.getOperand(1)->getName().str();
      if (new_to_old_qreg_names.count(seen_buf_name)) {
        seen_buf_name = new_to_old_qreg_names[seen_buf_name];
      }
      buffer_names.push_back(seen_buf_name);
      unique_buffer_names.insert(seen_buf_name);
    } else if (f && demangle(f->getName().str().c_str()).find("quantum::") !=
                        std::string::npos) {

      auto qrt_call_str = demangle(f->getName().str().c_str());
      auto split = xacc::split(qrt_call_str, ':');
      auto qrt_name = split[2].substr(0, split[2].find_first_of("("));

      if (qcor::qrt_to_xacc.count(qrt_name)) {
        auto xacc_name = qcor::qrt_to_xacc[qrt_name];
        auto inst = provider->createInstruction(xacc_name, seen_qbit_idxs);
        inst->setBufferNames(buffer_names);

        if (inst->nParameters() > 0) {
          xacc::InstructionParameter p;
          if (auto constant_double =
                  dyn_cast<ConstantFP>(invoke.getOperand(1))) {
            errs() << "Can get the double too "
                   << constant_double->getValueAPF().convertToDouble() << "\n";
            p = constant_double->getValueAPF().convertToDouble();
          } else {
            auto prev_node = invoke.getPrevNode();
            if (auto load = dyn_cast<LoadInst>(prev_node)) {

              // this was loading the parameter,
              // lets get the name as a string
              auto param_str = load->getOperand(0)->getName().str();
              param_str = xacc::split(param_str, '.')[0];
              errs() << "HELLO WORLD: " << param_str << "\n";
              p = param_str;
              composite->addVariable(param_str);
            }
          }

          inst->setParameter(0, p);
          //   exit(0);
        }
        seen_qbit_idxs.clear();
        buffer_names.clear();
        composite->addInstruction(inst);
      }
    }
  }
};

class XACC_To_LLVM_IR : public InstVisitor<XACC_To_LLVM_IR> {
protected:
  bool has_run_once = false;
  xacc::CompositeInstruction *program;
  Module *module;
  Function *simple_one_qbit;
  LLVMContext &context;

  std::map<std::string, StoreInst *> &variable_store_insts;

public:
  BasicBlock *basic_block;
  BasicBlock *execution_block;

  XACC_To_LLVM_IR(Module *mod, std::map<std::string, StoreInst *> &vsi,
                  xacc::CompositeInstruction *c)
      : program(c), module(mod), context(mod->getContext()),
        variable_store_insts(vsi) {}
  auto demangle(const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  }
  void visitBasicBlock(BasicBlock &bb) {
    // we are looking for if (__execute) block, so
    // looking for __execute load inst, should be second in the block
    if (bb.getInstList().size() > 1) {

      auto inst_iter = bb.getInstList().begin();
      inst_iter++;
      if (isa<LoadInst>(*inst_iter) &&
          demangle(dyn_cast<LoadInst>(&*inst_iter)
                       ->getOperand(0)
                       ->getName()
                       .str()
                       .c_str()) == "xacc::internal_compiler::__execute") {
        execution_block = &bb;
        Instruction *first_inst = &*bb.getInstList().begin();
        if (auto call = dyn_cast<CallInst>(first_inst)) {
          auto f = call->getCalledFunction();
          if (f &&
              demangle(f->getName().str().c_str())
                      .find("std::pair<std::__cxx11::basic_string<char, "
                            "std::char_traits<char>, std::allocator<char> >, "
                            "unsigned long>::~pair()") != std::string::npos) {
            call->eraseFromParent();
          }
        }
        return;
      }
    }
  }

  void visitInvokeInst(InvokeInst &invoke) {
    auto f = invoke.getCalledFunction();
    if (f && demangle(f->getName().str().c_str()).find("quantum::") !=
                 std::string::npos) {
      auto qrt_call_str = demangle(f->getName().str().c_str());
      auto split = xacc::split(qrt_call_str, ':');
      auto qrt_name = split[2].substr(0, split[2].find_first_of("("));
      if (qrt_to_xacc.count(qrt_name)) {
        if (!has_run_once) {
          auto normal_next = invoke.getNormalDest();
          auto except_next = invoke.getUnwindDest();
          Instruction *last_node = &invoke;

          Function *one_qubit = nullptr;
          Function *one_qubit_param = nullptr;
          Function *two_qubit = nullptr;
          // this is our first quantum call...
          for (int i = program->nInstructions() - 1; i >= 0; i--) {
            auto inst = program->getInstruction(i);
            // create call to qrt internal simple* call...
            int n_bits = inst->nRequiredBits(), n_params = inst->nParameters();

            if (inst->name() == "Measure") {
              n_params = 0;
            }

            IRBuilder<> builder(last_node->getParent());

            // create the gate name string
            Constant *gate_name = builder.CreateGlobalStringPtr(inst->name());
            // create the buffer register name string
            Constant *buf =
                builder.CreateGlobalStringPtr(inst->getBufferNames()[0]);

            if (n_bits == 1) {
              // If this is the first iteration, grab
              // the invoke inst's previous node and erase it
              // it corresponds to the qreg[IDX] call, we don't need it
              if (i == program->nInstructions() - 1) {
                last_node->getPrevNode()->eraseFromParent();
              }

              std::vector<Type *> arg_types_vec{
                  gate_name->getType(), gate_name->getType(),
                  FunctionType::getInt64Ty(context)};
              std::vector<Value *> args_vec{
                  gate_name, buf,
                  ConstantInt::get(IntegerType::getInt64Ty(context),
                                   APInt(64, inst->bits()[0]))};
              if (n_params > 0) {
                // rotation gate
                for (auto &p : inst->getParameters()) {

                  // add to the arg types
                  arg_types_vec.push_back(FunctionType::getDoubleTy(context));

                  if (p.isVariable()) {

                    // this parameter string should correspond to
                    // and argument on the function
                    // TODO Add a LoadInst to load VARNAME.addr
                    // then add that return value to args.
                    //   %0 = load double, double* %angle.addr, align 8
                    //  LoadInst(Type *Ty, Value *Ptr, const Twine &NameStr,
                    //  bool isVolatile,
                    //    Instruction *InsertBefore = nullptr);
                    auto store_inst_key = p.toString() + ".addr";
                    auto load = new LoadInst(
                        FunctionType::getDoubleTy(context),
                        variable_store_insts[store_inst_key]->getOperand(1),
                        "tmp_" + p.toString(), false, last_node);
                    auto load_value = load->getOperand(0);
                    args_vec.push_back(load);
                  } else {
                    args_vec.push_back(ConstantFP::get(
                        FunctionType::getDoubleTy(context),
                        APFloat(xacc::InstructionParameterToDouble(p))));
                  }
                }
              }

              ArrayRef<Type *> arg_types(arg_types_vec);
              ArrayRef<Value *> args(args_vec);

              FunctionType *ftype = FunctionType::get(
                  FunctionType::getVoidTy(context), arg_types, false);

              Instruction *new_inst = nullptr;

              if (n_params > 0) {
                if (!one_qubit_param) {
                  one_qubit_param = Function::Create(
                      ftype, Function::ExternalLinkage,
                      "_ZN4xacc17internal_compiler38simplified_qrt_call_one_"
                      "qbit_one_paramEPKcS2_md",
                      module);
                }

                // Create the call inst and add it to the Function
                if (i == program->nInstructions() - 1) {
                  new_inst =
                      InvokeInst::Create(ftype, one_qubit_param, normal_next,
                                         except_next, args, "", last_node);
                } else {
                  new_inst = CallInst::Create(ftype, one_qubit_param, args, "",
                                              last_node);
                }
              } else {
                if (!one_qubit) {
                  one_qubit = Function::Create(
                      ftype, Function::ExternalLinkage,
                      "_ZN4xacc17internal_compiler28simplified_qrt_call_"
                      "one_qbitEPKcS2_m",
                      module);
                }

                // Create the call inst and add it to the Function
                if (i == program->nInstructions() - 1) {
                  new_inst =
                      InvokeInst::Create(ftype, one_qubit, normal_next,
                                         except_next, args, "", last_node);
                } else {
                  new_inst =
                      CallInst::Create(ftype, one_qubit, args, "", last_node);
                }
              }

              last_node = new_inst;

              if (i == program->nInstructions() - 1) {
                // save this basic block...
                basic_block = invoke.getParent();
                invoke.eraseFromParent();
              }

            } else {
              // 2 qubit gates
              if (n_params > 0) {
                  // TODO FIXME add 2 qubit gates with param
              } else {
                // _ZN4xacc17internal_compiler29simplified_qrt_call_two_qbitsEPKcS2_S2_mm

                // create the buffer register name string
                Constant *buf_2 =
                    builder.CreateGlobalStringPtr(inst->getBufferNames()[1]);

                // set the argument Types for this function call (char *,
                // char *, size_t)
                ArrayRef<Type *> arg_types{
                    gate_name->getType(), gate_name->getType(),
                    gate_name->getType(), FunctionType::getInt64Ty(context),
                    FunctionType::getInt64Ty(context)};

                // void return type, create the FunctionType instance
                FunctionType *ftype = FunctionType::get(
                    FunctionType::getVoidTy(context), arg_types, false);

                if (!two_qubit) {
                  two_qubit = Function::Create(
                      ftype, Function::ExternalLinkage,
                      "_ZN4xacc17internal_compiler29simplified_qrt_call_"
                      "two_qbitsEPKcS2_S2_mm",
                      module);
                }
                // create actual argument values
                ArrayRef<Value *> args{
                    gate_name, buf, buf_2,
                    ConstantInt::get(IntegerType::getInt64Ty(context),
                                     APInt(64, inst->bits()[0])),
                    ConstantInt::get(IntegerType::getInt64Ty(context),
                                     APInt(64, inst->bits()[1]))};

                // Create the call inst and add it to the Function
                Instruction *new_inst;
                if (i == program->nInstructions() - 1) {
                  new_inst =
                      InvokeInst::Create(ftype, two_qubit, normal_next,
                                         except_next, args, "", last_node);
                } else {
                  new_inst =
                      CallInst::Create(ftype, two_qubit, args, "", last_node);
                }

                last_node = new_inst;

                if (i == program->nInstructions() - 1) {
                  // save this basic block...
                  basic_block = invoke.getParent();
                  invoke.eraseFromParent();
                }
              }
            }
          }

          has_run_once = true;
        } else {
          auto containing_bb = invoke.getParent();

          int n_bits = xacc::getIRProvider("quantum")->getNRequiredBits(
              qrt_to_xacc[qrt_name]);
          if (n_bits == 1) {
            auto node = invoke.getPrevNode();

            if (isa<LoadInst>(node)) {
              // this is only for single parameter gates
              // this means that we have load on a gate parameter
              node = node->getPrevNode();
            }
            node->eraseFromParent();

          } else if (n_bits == 2) {
            // invoke.getPrevNode()->getPrevNode()->eraseFromParent();
            // invoke.getPrevNode()->eraseFromParent();
            containing_bb->getPrevNode();
          }
          invoke.eraseFromParent();

          if (!containing_bb->hasNPredecessorsOrMore(1)) {
            containing_bb->eraseFromParent();
          }
        }
      }
    }
  }
};

class SearchForCallsToThisFunction
    : public InstVisitor<SearchForCallsToThisFunction> {

protected:
  Function *function;

public:
  CallInst *found_call = nullptr;

  SearchForCallsToThisFunction(Function *f) : function(f) {}
  auto demangle(const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  }
  void visitCallInst(CallInst &call) {
    if (call.getCalledFunction() != nullptr) {
      //   errs() <<
      //   demangle(call.getCalledFunction()->getName().str().c_str()) <<
      //   "\n";
      if (call.getCalledFunction() == function) {
        errs() << "Call Found our function\n";
        call.dump();
        call.getParent()->getParent();
        found_call = &call;
      }
    }
  }
  void visitInvokeInst(InvokeInst &call) {
    if (call.getCalledFunction() != nullptr) {
      //   errs() <<
      //   demangle(call.getCalledFunction()->getName().str().c_str()) <<
      //   "\n";
      if (call.getCalledFunction() == function) {
        errs() << "Invoke Found our function\n";
        call.dump();
      }
    }
  }
};
} // namespace qcor