#include "ModifierRegionLowering.hpp"

#include <iostream>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace {
// Inline a region into a location specified by an Op
void inlineRegion(mlir::Region *regionToInline,
                  mlir::Operation *inlineLocation) {
  mlir::Block *insertBlock = inlineLocation->getBlock();
  assert(insertBlock);
  mlir::Region *insertRegion = insertBlock->getParent();
  // Split the insertion block.
  mlir::Block *postInsertBlock =
      insertBlock->splitBlock(inlineLocation->getIterator());
  mlir::BlockAndValueMapping mapper;
  regionToInline->cloneInto(insertRegion, postInsertBlock->getIterator(),
                            mapper);
  auto newBlocks = llvm::make_range(std::next(insertBlock->getIterator()),
                                    postInsertBlock->getIterator());
  mlir::Block *firstNewBlock = &*newBlocks.begin();

  auto *firstBlockTerminator = firstNewBlock->getTerminator();
  firstBlockTerminator->erase();
  // Merge the post insert block into the cloned entry block.
  firstNewBlock->getOperations().splice(firstNewBlock->end(),
                                        postInsertBlock->getOperations());
  postInsertBlock->erase();
  // Splice the instructions of the inlined entry block into the insert block.
  insertBlock->getOperations().splice(insertBlock->end(),
                                      firstNewBlock->getOperations());
  firstNewBlock->erase();
}
} // namespace
namespace qcor {
// Note: the Modifier regions implement quantum dataflow analysis (value
// semantics) by returning new mlir::Value of all qubit operands (to be
// continued by subsequent QVS Ops) At this lowering stage, we make the
// connection b/w the SSA values from inside the region to the outside scope:
// For example:
//     %33 = q.pow(%c2_i64) {
//       %37 = qvs.x(%9) : !quantum.Qubit
//       "quantum.modifier_end"(%37) : (!quantum.Qubit) -> ()
//     }
// We connect %33 and %37 (operands of the modifier block terminator).
LogicalResult PowURegionOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto location = parentModule->getLoc();
  // End
  {
    FlatSymbolRefAttr qir_get_fn_ptr = [&]() {
      static const std::string qir_end_func = "__quantum__rt__end_pow_u_region";
      if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_end_func)) {
        return SymbolRefAttr::get(qir_end_func, context);
      } else {
        // prototype should be (int64) -> void :

        auto void_type = LLVM::LLVMVoidType::get(context);

        auto func_type = LLVM::LLVMFunctionType::get(
            void_type, llvm::ArrayRef<Type>{rewriter.getI64Type()}, false);

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(location, qir_end_func, func_type);

        return mlir::SymbolRefAttr::get(qir_end_func, context);
      }
    }();

    // Only forward the first operand (pow) to __quantum__rt__end_pow_u_region
    const std::vector<mlir::Value> qir_operands{operands[0]};
    rewriter.create<mlir::CallOp>(location, qir_get_fn_ptr,
                                  LLVM::LLVMVoidType::get(context),
                                  qir_operands);
  }

  {
    auto casted = cast<mlir::quantum::PowURegion>(op);
    mlir::SmallVector<mlir::Value> chained_values;
    for (const auto &targetQubit : casted.qubits()) {
      chained_values.push_back(targetQubit);
    }
    rewriter.replaceOp(op, chained_values);
  }

  return success();
}

LogicalResult CtrlURegionOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto location = parentModule->getLoc();
  // End
  {
    FlatSymbolRefAttr qir_get_fn_ptr = [&]() {
      static const std::string qir_end_func =
          "__quantum__rt__end_ctrl_u_region";
      if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_end_func)) {
        return SymbolRefAttr::get(qir_end_func, context);
      } else {
        // prototype should be (Qubit * ) -> void :
        // ret is void
        auto void_type = LLVM::LLVMVoidType::get(context);

        auto func_type = LLVM::LLVMFunctionType::get(
            void_type,
            llvm::ArrayRef<Type>{
                LLVM::LLVMPointerType::get(get_quantum_type("Qubit", context))},
            false);

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(location, qir_end_func, func_type);

        return mlir::SymbolRefAttr::get(qir_end_func, context);
      }
    }();

    mlir::Value ctrl_bit = operands[0];
    if (auto q_op =
            ctrl_bit.getDefiningOp<mlir::quantum::ValueSemanticsInstOp>()) {
      ctrl_bit = q_op.getOperands()[0];
    }

    rewriter.create<mlir::CallOp>(
        location, qir_get_fn_ptr, LLVM::LLVMVoidType::get(context),
        llvm::makeArrayRef(std::vector<mlir::Value>{ctrl_bit}));
  }

  {
    auto casted = cast<mlir::quantum::CtrlURegion>(op);
    mlir::SmallVector<mlir::Value> chained_values{casted.ctrl_qubit()};
    for (const auto &targetQubit : casted.qubits()) {
      chained_values.push_back(targetQubit);
    }
    rewriter.replaceOp(op, chained_values);
  }
  return success();
}

LogicalResult AdjURegionOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Local Declarations
  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
  auto context = parentModule->getContext();
  auto location = parentModule->getLoc();
  // End
  {
    FlatSymbolRefAttr qir_get_fn_ptr = [&]() {
      static const std::string qir_end_func = "__quantum__rt__end_adj_u_region";
      if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_end_func)) {
        return SymbolRefAttr::get(qir_end_func, context);
      } else {
        // prototype should be () -> void :

        auto void_type = LLVM::LLVMVoidType::get(context);

        auto func_type = LLVM::LLVMFunctionType::get(
            void_type, llvm::ArrayRef<Type>{}, false);

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(parentModule.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(location, qir_end_func, func_type);

        return mlir::SymbolRefAttr::get(qir_end_func, context);
      }
    }();

    rewriter.create<mlir::CallOp>(
        location, qir_get_fn_ptr, LLVM::LLVMVoidType::get(context),
        llvm::makeArrayRef(std::vector<mlir::Value>{}));
  }

  {
    mlir::SmallVector<mlir::Value> chained_values;
    for (const auto &targetQubit :
         cast<mlir::quantum::AdjURegion>(op).qubits()) {
      chained_values.push_back(targetQubit);
    }
    rewriter.replaceOp(op, chained_values);
  }

  return success();
}

LogicalResult EndModifierRegionOpLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  // Just erase
  rewriter.eraseOp(op);
  return success();
}

void ModifierRegionRewritePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void ModifierRegionRewritePass::runOnOperation() {
  const auto insertStartCall = [](const std::string &qir_start_func,
                                  mlir::OpBuilder &opBuilder,
                                  mlir::ModuleOp &parentModule) {
    mlir::FlatSymbolRefAttr startModifiedU = [&]() {
      PatternRewriter::InsertionGuard insertGuard(opBuilder);
      opBuilder.setInsertionPointToStart(
          &parentModule.getRegion().getBlocks().front());
      if (parentModule.lookupSymbol<mlir::FuncOp>(qir_start_func)) {
        auto fnNameAttr = opBuilder.getSymbolRefAttr(qir_start_func);
        return fnNameAttr;
      }

      auto func_decl = opBuilder.create<mlir::FuncOp>(
          opBuilder.getUnknownLoc(), qir_start_func,
          opBuilder.getFunctionType(llvm::None, llvm::None));
      func_decl.setVisibility(mlir::SymbolTable::Visibility::Private);
      return mlir::SymbolRefAttr::get(qir_start_func,
                                      parentModule->getContext());
    }();

    opBuilder.create<mlir::CallOp>(opBuilder.getUnknownLoc(), startModifiedU,
                                   llvm::None, llvm::None);
  };

  getOperation().walk([&](mlir::quantum::AdjURegion op) {
    mlir::OpBuilder rewriter(op);
    mlir::ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    insertStartCall("__quantum__rt__start_adj_u_region", rewriter,
                    parentModule);
    inlineRegion(&op.body(), op.getOperation());
  });

  getOperation().walk([&](mlir::quantum::CtrlURegion op) {
    mlir::OpBuilder rewriter(op);
    mlir::ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    insertStartCall("__quantum__rt__start_ctrl_u_region", rewriter,
                    parentModule);
    inlineRegion(&op.body(), op.getOperation());
  });

  getOperation().walk([&](mlir::quantum::PowURegion op) {
    mlir::OpBuilder rewriter(op);
    mlir::ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    insertStartCall("__quantum__rt__start_pow_u_region", rewriter,
                    parentModule);
    inlineRegion(&op.body(), op.getOperation());
  });
}
} // namespace qcor