#include "ModifierBlockInliner.hpp"
#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>
#include <unordered_map>
namespace qcor {
void ModifierBlockInlinerPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void ModifierBlockInlinerPass::handlePowU() {
  // Power U to loop...
  std::vector<Operation *> deadOps;

  getOperation().walk([&](mlir::quantum::PowURegion op) {
    // Must be a single-block op
    assert(op.body().getBlocks().size() == 1);
    mlir::OpBuilder rewriter(op);
    assert(op.pow().getType().isIndex());
    mlir::Value powVal = op.pow();
    if (!mlir::isValidDim(powVal)) {
      return;
    }
    mlir::Value lbs_val = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    mlir::Block &powBlock = op.body().getBlocks().front();
    // Convert the pow modifier to a For loop,
    // which might be unrolled if possible (constant-value loop bound)
    mlir::ValueRange lbs(lbs_val);
    mlir::ValueRange ubs(powVal);
    rewriter.create<mlir::AffineForOp>(
        op.getLoc(), lbs, rewriter.getMultiDimIdentityMap(lbs.size()), ubs,
        rewriter.getMultiDimIdentityMap(ubs.size()), 1, llvm::None,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::Value iv, mlir::ValueRange itrArgs) {
          mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
          mlir::BlockAndValueMapping mapper;
          for (auto &subOp : powBlock.getOperations()) {
            auto newOp = nestedBuilder.clone(subOp, mapper);
            if (auto terminator =
                    mlir::dyn_cast_or_null<mlir::quantum::ModifierEndOp>(
                        newOp)) {
              nestedBuilder.create<mlir::AffineYieldOp>(nestedLoc);
              newOp->erase();
              break;
            }
          }
        });

    for (size_t i = 0; i < op.result().size(); ++i) {
      op.result()[i].replaceAllUsesExcept(
          op.qubits()[i], mlir::SmallPtrSet<Operation *, 1>{op});
    }
    op.body().getBlocks().clear();
    deadOps.emplace_back(op.getOperation());
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();
}

void ModifierBlockInlinerPass::applyControlledQuantumOp(
    mlir::quantum::ValueSemanticsInstOp &qvsOp, mlir::Value control_bit,
    mlir::OpBuilder &rewriter) {
  const auto inst_name = qvsOp.name();
  auto& op = qvsOp;
  if (inst_name == "x") {
    std::vector<mlir::Type> ret_types{qvsOp.getOperand(0).getType(),
                                      qvsOp.getOperand(0).getType()};
    std::vector<mlir::Value> qubit_operands{control_bit, qvsOp.getOperand(0)};
    auto new_inst = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        qvsOp.getLoc(), llvm::makeArrayRef(ret_types), "cx",
        llvm::makeArrayRef(qubit_operands), llvm::None);
    mlir::Value new_ctrl_qubit_ssa = new_inst.result().front();
    control_bit.replaceAllUsesExcept(
        new_ctrl_qubit_ssa, mlir::SmallPtrSet<Operation *, 1>{new_inst});
    // Update the target qubit SSA use-def:
    qvsOp.result()[0].replaceAllUsesExcept(
        new_inst.result().back(), mlir::SmallPtrSet<Operation *, 1>{new_inst});
  } else if (inst_name == "y") {
    // cy a,b { sdg b; cx a,b; s b; }
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    auto sdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        op.getLoc(), llvm::makeArrayRef({qubit_type}), "sdg",
        llvm::makeArrayRef({b}), llvm::None);
    // !IMPORTANT! track use-def chain as well
    b = sdg.result().front();
    auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
        llvm::makeArrayRef({a, b}), llvm::None);
    a = cx.result().front();
    b = cx.result().back();
    auto s = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        op.getLoc(), llvm::makeArrayRef({qubit_type}), "s",
        llvm::makeArrayRef({b}), llvm::None);
    b = s.result().front();
    control_bit.replaceAllUsesExcept(
        a, mlir::SmallPtrSet<Operation *, 3>{sdg, cx, s});
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "z") {
    // cz a,b { h b; cx a,b; h b; }
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    {
      // h b
      auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
          llvm::makeArrayRef({b}), llvm::None);
      b = h.getResults().front();
      newOps.emplace_back(h);
    }
    {
      // cx a,b
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // h b
      auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
          llvm::makeArrayRef({b}), llvm::None);
      b = h.getResults().front();
      newOps.emplace_back(h);
    }

    mlir::SmallPtrSet<Operation *, 3> newOpPtrs;
    for (auto &x : newOps) {
      newOpPtrs.insert(x);
    }
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "h") {
    // gate ch a,b {
    //   h b; sdg b;
    //   cx a,b;
    //   h b; t b;
    //   cx a,b;
    //   t b; h b; s b; x b; s a;
    //   }
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    {
      // h b
      auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
          llvm::makeArrayRef({b}), llvm::None);
      b = h.getResults().front();
      newOps.emplace_back(h);
    }
    {
      // sdg b
      auto sdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "sdg",
          llvm::makeArrayRef({b}), llvm::None);
      b = sdg.result().front();
      newOps.emplace_back(sdg);
    }
    {
      // cx a,b
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // h b
      auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
          llvm::makeArrayRef({b}), llvm::None);
      b = h.getResults().front();
      newOps.emplace_back(h);
    }
    {
      // t b
      auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
          llvm::makeArrayRef({b}), llvm::None);
      b = t.getResults().front();
      newOps.emplace_back(t);
    }
    {
      // cx a,b
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // t b
      auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
          llvm::makeArrayRef({b}), llvm::None);
      b = t.getResults().front();
      newOps.emplace_back(t);
    }
    {
      // h b
      auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
          llvm::makeArrayRef({b}), llvm::None);
      b = h.getResults().front();
      newOps.emplace_back(h);
    }
    {
      // s b
      auto s = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "s",
          llvm::makeArrayRef({b}), llvm::None);
      b = s.getResults().front();
      newOps.emplace_back(s);
    }
    {
      // x b
      auto x = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "x",
          llvm::makeArrayRef({b}), llvm::None);
      b = x.getResults().front();
      newOps.emplace_back(x);
    }
    {
      // s a
      auto s = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "s",
          llvm::makeArrayRef({a}), llvm::None);
      a = s.getResults().front();
      newOps.emplace_back(s);
    }
    assert(newOps.size() == 11);
    mlir::SmallPtrSet<Operation *, 11> newOpPtrs;
    for (auto &x : newOps) {
      newOpPtrs.insert(x);
    }
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "t") {
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    // Ctrl-T = CPhase(pi/4)
    mlir::Value pi_over_4 = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), M_PI / 4));
    auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cphase",
        llvm::makeArrayRef({a, b}), llvm::makeArrayRef({pi_over_4}));
    a = cp.getResults().front();
    b = cp.getResults().back();
    mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "tdg") {
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    // Ctrl-Tdg = CPhase(-pi/4)
    mlir::Value minus_pi_over_4 = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -M_PI / 4));
    auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cphase",
        llvm::makeArrayRef({a, b}), llvm::makeArrayRef({minus_pi_over_4}));
    a = cp.getResults().front();
    b = cp.getResults().back();
    mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "s") {
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    // Ctrl-S = CPhase(pi/2)
    mlir::Value pi_over_2 = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), M_PI / 2));
    auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cphase",
        llvm::makeArrayRef({a, b}), llvm::makeArrayRef({pi_over_2}));
    a = cp.getResults().front();
    b = cp.getResults().back();
    mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "sdg") {
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    // Ctrl-Sdg = CPhase(-pi/2)
    mlir::Value minus_pi_over_2 = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -M_PI / 2));
    auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
        op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cphase",
        llvm::makeArrayRef({a, b}), llvm::makeArrayRef({minus_pi_over_2}));
    a = cp.getResults().front();
    b = cp.getResults().back();
    mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "rx") {
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;

    mlir::Value lambda = qvsOp.getOperand(1);
    mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
    mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
    mlir::Value lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
    mlir::Value minus_lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_minus_two);

    {
      // rx(lambda/2) b
      auto rx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "rx",
          llvm::makeArrayRef({b}), llvm::makeArrayRef({lambda_over_2}));
      b = rx.getResults().front();
      newOps.emplace_back(rx);
    }
    {
      // cx a,b
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // rx(-lambda/2) b
      auto rx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "rx",
          llvm::makeArrayRef({b}), llvm::makeArrayRef({minus_lambda_over_2}));
      b = rx.getResults().front();
      newOps.emplace_back(rx);
    }
    {
      // cx a,b
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }

    mlir::SmallPtrSet<Operation *, 4> newOpPtrs;
    for (auto &x : newOps) {
      newOpPtrs.insert(x);
    }
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "ry") {
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    mlir::Value lambda = qvsOp.getOperand(1);
    mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
    mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
    mlir::Value lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
    mlir::Value minus_lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_minus_two);
    // gate cry(lambda) a,b
    // {
    //   ry(lambda/2) b;
    //   cx a,b;
    //   ry(-lambda/2) b;
    //   cx a,b;
    // }
    {
      // ry(lambda/2) b;
      auto ry = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "ry",
          llvm::makeArrayRef({b}), llvm::makeArrayRef({lambda_over_2}));
      b = ry.getResults().front();
      newOps.emplace_back(ry);
    }
    {
      // cx a,b;
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // ry(-lambda/2) b;
      auto ry = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "ry",
          llvm::makeArrayRef({b}), llvm::makeArrayRef({minus_lambda_over_2}));
      b = ry.getResults().front();
      newOps.emplace_back(ry);
    }
    {
      // cx a,b;
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }

    mlir::SmallPtrSet<Operation *, 4> newOpPtrs;
    for (auto &x : newOps) {
      newOpPtrs.insert(x);
    }
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "rz") {
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    mlir::Value lambda = qvsOp.getOperand(1);
    mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
    mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
    mlir::Value lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
    mlir::Value minus_lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_minus_two);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    // gate crz(lambda) a,b
    // {
    //   rz(lambda/2) b;
    //   cx a,b;
    //   rz(-lambda/2) b;
    //   cx a,b;
    // }
    {
      // rz(lambda/2) b
      auto rz = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "rz",
          llvm::makeArrayRef({b}), llvm::makeArrayRef({lambda_over_2}));
      b = rz.getResults().front();
      newOps.emplace_back(rz);
    }
    {
      // cx a,b
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // rz(-lambda/2) b
      auto rz = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "rz",
          llvm::makeArrayRef({b}), llvm::makeArrayRef({minus_lambda_over_2}));
      b = rz.getResults().front();
      newOps.emplace_back(rz);
    }
    {
      // cx a,b
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }

    mlir::SmallPtrSet<Operation *, 4> newOpPtrs;
    for (auto &x : newOps) {
      newOpPtrs.insert(x);
    }
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
  } else if (inst_name == "cx" || inst_name == "cnot") {
    // gate ccx a,b,c
    // {
    //   h c;
    //   cx b,c; tdg c;
    //   cx a,c; t c;
    //   cx b,c; tdg c;
    //   cx a,c; t b; t c; h c;
    //   cx a,b; t a; tdg b;
    //   cx a,b;
    // }
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    mlir::Value c = qvsOp.getOperand(1);
    // h c;
    {
      auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
          llvm::makeArrayRef({c}), llvm::None);
      c = h.getResults().front();
      newOps.emplace_back(h);
    }
    // cx b,c
    {
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({b, c}), llvm::None);
      b = cx.getResults().front();
      c = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    // tdg c;
    {
      auto tdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "tdg",
          llvm::makeArrayRef({c}), llvm::None);
      c = tdg.getResults().front();
      newOps.emplace_back(tdg);
    }
    // cx a,c
    {
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, c}), llvm::None);
      a = cx.getResults().front();
      c = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    // t c
    {
      auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
          llvm::makeArrayRef({c}), llvm::None);
      c = t.getResults().front();
      newOps.emplace_back(t);
    }
    // cx b,c
    {
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({b, c}), llvm::None);
      b = cx.getResults().front();
      c = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    // tdg c;
    {
      auto tdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "tdg",
          llvm::makeArrayRef({c}), llvm::None);
      c = tdg.getResults().front();
      newOps.emplace_back(tdg);
    }
    // cx a,c
    {
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, c}), llvm::None);
      a = cx.getResults().front();
      c = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    // t b
    {
      auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
          llvm::makeArrayRef({b}), llvm::None);
      b = t.getResults().front();
      newOps.emplace_back(t);
    }
    // t c
    {
      auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
          llvm::makeArrayRef({c}), llvm::None);
      c = t.getResults().front();
      newOps.emplace_back(t);
    }
    // h c;
    {
      auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
          llvm::makeArrayRef({c}), llvm::None);
      c = h.getResults().front();
      newOps.emplace_back(h);
    }
    // cx a,b
    {
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    // t a
    {
      auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
          llvm::makeArrayRef({a}), llvm::None);
      a = t.getResults().front();
      newOps.emplace_back(t);
    }
    // tdg b;
    {
      auto tdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type}), "tdg",
          llvm::makeArrayRef({b}), llvm::None);
      b = tdg.getResults().front();
      newOps.emplace_back(tdg);
    }
    // cx a,b;
    {
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({a, b}), llvm::None);
      a = cx.getResults().front();
      b = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    assert(newOps.size() == 15);
    mlir::SmallPtrSet<Operation *, 15> newOpPtrs;
    for (auto &x : newOps) {
      newOpPtrs.insert(x);
    }
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
    qvsOp.getResults().back().replaceAllUsesWith(c);
  } else if (inst_name == "cphase") {
    // Ref:
    // ccu1(lambda, a, b, c) =
    // cu1(lambda/2, a, b)
    // cx(b, c)
    // cu1(-lambda/2, a, c)
    // cx(b, c)
    // cu1(lambda/2, a, c)
    mlir::Type qubit_type = qvsOp.getOperand(0).getType();
    mlir::Value a = control_bit;
    mlir::Value b = qvsOp.getOperand(0);
    mlir::Value c = qvsOp.getOperand(1);
    mlir::Value lambda = qvsOp.getOperand(2);
    mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
    mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
    mlir::Value lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
    mlir::Value minus_lambda_over_2 =
        rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_minus_two);
    std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
    {
      // cu1(lambda/2, a, b)
      auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cphase",
          llvm::makeArrayRef({a, b}), llvm::makeArrayRef({lambda_over_2}));
      a = cp.getResults().front();
      b = cp.getResults().back();
      newOps.emplace_back(cp);
    }
    {
      // cx(b, c)
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({b, c}), llvm::None);
      b = cx.getResults().front();
      c = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // cu1(-lambda/2, a, c)
      auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cphase",
          llvm::makeArrayRef({a, c}),
          llvm::makeArrayRef({minus_lambda_over_2}));
      a = cp.getResults().front();
      c = cp.getResults().back();
      newOps.emplace_back(cp);
    }
    {
      // cx(b, c)
      auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
          llvm::makeArrayRef({b, c}), llvm::None);
      b = cx.getResults().front();
      c = cx.getResults().back();
      newOps.emplace_back(cx);
    }
    {
      // cu1(lambda/2, a, c)
      auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
          op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cphase",
          llvm::makeArrayRef({a, c}), llvm::makeArrayRef({lambda_over_2}));
      a = cp.getResults().front();
      c = cp.getResults().back();
      newOps.emplace_back(cp);
    }
    assert(newOps.size() == 5);
    mlir::SmallPtrSet<Operation *, 5> newOpPtrs;
    for (auto &x : newOps) {
      newOpPtrs.insert(x);
    }
    control_bit.replaceAllUsesExcept(a, newOpPtrs);
    qvsOp.getResults().front().replaceAllUsesWith(b);
    qvsOp.getResults().back().replaceAllUsesWith(c);
  } else {
    // We don't expect this gate just yet, need to add.
    std::cout << "Unknown quantum gate: " << inst_name.str() << "\n";
    assert(false);
  }
}

void ModifierBlockInlinerPass::handleCtrlU() {
  std::function<bool(mlir::Operation &)> is_quantum_op =
      [&is_quantum_op](mlir::Operation &opToCheck) -> bool {
    if (mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(
            &opToCheck)) {
      return true;
    }
    if (opToCheck.getNumRegions() > 0) {
      for (auto &subRegion : opToCheck.getRegions()) {
        for (auto &subBlock : subRegion.getBlocks()) {
          for (auto &subOp : subBlock.getOperations()) {
            // Recurse
            if (is_quantum_op(subOp)) {
              return true;
            }
          }
        }
      }
    }

    return false;
  };

  // Control-U
  std::vector<Operation *> deadOps;
  getOperation().walk([&](mlir::quantum::CtrlURegion op) {
    // Must be a single-block op
    assert(op.body().getBlocks().size() == 1);
    mlir::OpBuilder rewriter(op);
    mlir::Block &ctrlBlock = op.body().getBlocks().front();
    for (auto &subOp : ctrlBlock.getOperations()) {
      if (mlir::dyn_cast_or_null<mlir::quantum::ModifierEndOp>(&subOp)) {
        break;
      }
      // Limit the auto ctrl-gate auto gen to sequence of gates only atm.
      // TODO: The inline MLIR tree modification procedure (wrapping ops into
      // new regions, etc.) is not robust for all cases.
      if (!is_quantum_op(subOp)) {
        return;
      }
    }

    for (auto &subOp : ctrlBlock.getOperations()) {
      // We're at the end
      if (auto terminator =
              mlir::dyn_cast_or_null<mlir::quantum::ModifierEndOp>(&subOp)) {
        for (size_t i = 0; i < terminator.qubits().size(); ++i) {
          op.result()[i].replaceAllUsesWith(terminator.qubits()[i]);
        }
        break;
      }

      // this is not a quantum op:
      if (!is_quantum_op(subOp)) {
        rewriter.insert(subOp.clone());
        continue;
      }
      if (mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(&subOp)) {
        // Simple gates:
        mlir::quantum::ValueSemanticsInstOp qvsOp =
            mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(&subOp);
        applyControlledQuantumOp(qvsOp, op.ctrl_qubit(), rewriter);
      } else {
        // Complex cases: for now, just put in the ctrl block.
        // i.e., runtime to handle.
        auto ctrlUOp =
          rewriter.create<mlir::quantum::CtrlURegion>(op.getLoc(), op.ctrl_qubit(), llvm::None);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&ctrlUOp.body().front());
          rewriter.insert(subOp.clone());
          rewriter.create<mlir::quantum::ModifierEndOp>(op.getLoc(), llvm::None);
        }
      }
    }
    op.body().getBlocks().clear();
    deadOps.emplace_back(op.getOperation());
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();
}
void ModifierBlockInlinerPass::handleAdjU() {
  std::vector<Operation *> deadOps;
  getOperation().walk([&](mlir::quantum::AdjURegion op) {
    // Must be a single-block op
    assert(op.body().getBlocks().size() == 1);
    mlir::OpBuilder rewriter(op);
    mlir::Block &adjBlock = op.body().getBlocks().front();
    // The only case that we handle now is that the adjoint block
    // is a linear sequence of quantum gates, nothing else.
    const bool isSimpleListOfQvsOps = [&]() {
      for (auto &subOp : adjBlock.getOperations()) {
        // see the terminator
        if (mlir::dyn_cast_or_null<mlir::quantum::ModifierEndOp>(&subOp)) {
          return true;
        }
        if (!mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(
                &subOp)) {
          return false;
        }
      }
      // something wrong here, e.g., missing terminator...
      assert(false);
      return false;
    }();
    // We can only flatten the Adjoint region in this case.
    // Otherwise, the adjoint region persists to LLVM lowering -> runtime handling.
    if (isSimpleListOfQvsOps) {
      std::vector<mlir::quantum::ValueSemanticsInstOp> opsToReverse;
      mlir::Operation *modifierEndOp = nullptr;
      for (auto &subOp : adjBlock.getOperations()) {
        // We're at the end
        if (mlir::dyn_cast_or_null<mlir::quantum::ModifierEndOp>(&subOp)) {
          modifierEndOp = &subOp;
          break;
        }

        // cast since we know the op must be a ValueSemanticsInstOp op:
        mlir::quantum::ValueSemanticsInstOp qvsOp =
            mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(&subOp);
        opsToReverse.emplace_back(qvsOp);
      }
      // Reverse the gate sequence
      // adjusting the SSA use-def chain accordingly.
      // We must do track the SSA chain backward.
      // List of all use-def chains
      std::vector<std::vector<mlir::Value>> use_def_chains;
      // Map from the opaque pointer to the chain index
      std::unordered_map<void*, size_t> ssa_to_chain_idx;
      for (auto& op: opsToReverse) {
        for (size_t q = 0; q < op.qubits().size(); ++q) {
          mlir::Value qubitOperand = op.qubits()[q];
          auto iter = ssa_to_chain_idx.find(qubitOperand.getAsOpaquePointer());
          if (iter != ssa_to_chain_idx.end()) {
            // This qubit line was generated internally
            const auto chainIdx = iter->second;
            assert(chainIdx < use_def_chains.size());
            auto& useDefList = use_def_chains[chainIdx];
            mlir::Value resultQubit = op.result()[q];
            // Add the result qubit to the tracking:
            assert(ssa_to_chain_idx.find(resultQubit.getAsOpaquePointer()) ==
                   ssa_to_chain_idx.end());
            ssa_to_chain_idx[resultQubit.getAsOpaquePointer()] = chainIdx;
            useDefList.emplace_back(resultQubit);
          } else {
            // First time seeing this use-def chain
            mlir::Value resultQubit = op.result()[q];
            use_def_chains.emplace_back(
                std::vector<mlir::Value>{qubitOperand, resultQubit});
            const auto chainIdx = use_def_chains.size() - 1;
            // Add both the input and result to the list
            ssa_to_chain_idx[qubitOperand.getAsOpaquePointer()] = chainIdx;
            ssa_to_chain_idx[resultQubit.getAsOpaquePointer()] = chainIdx;
          }
        }
      }

      
      std::unordered_map<void*, mlir::Value> qubit_operand_mapping;
      std::unordered_map<void*, mlir::Value> output_qubit_operand_mapping;
      for (auto &chain : use_def_chains) {
        assert(chain.size() >= 2);
        qubit_operand_mapping[chain.back().getAsOpaquePointer()] =
            chain.front();
      }
      // Now doing the reverse:
      const auto createInvGate = [&](mlir::quantum::ValueSemanticsInstOp
                                         &originalOp) {
        const std::string inst_name = originalOp.name().str();
        const std::unordered_map<std::string, std::string> GATE_MAPPING{
            {"x", "x"},   {"y", "y"},           {"z", "z"},   {"h", "h"},
            {"cx", "cx"}, {"cnot", "cnot"},     {"rx", "rx"}, {"ry", "ry"},
            {"rz", "rz"}, {"cphase", "cphase"}, {"t", "tdg"}, {"tdg", "t"},
            {"s", "sdg"}, {"sdg", "s"}};
        assert(GATE_MAPPING.find(inst_name) != GATE_MAPPING.end());
        const std::string invGateName = GATE_MAPPING.find(inst_name)->second;
        const size_t nQubits = originalOp.qubits().size();
        assert(nQubits == 1 || nQubits == 2);
        const bool isParameterized = !originalOp.params().empty();
        if (!isParameterized) {
          if (nQubits == 1) {
            mlir::Value inputQubit = originalOp.getOperand(0);
            mlir::Value outputQubit = originalOp.getResult(0);
            mlir::Type qubit_type = originalOp.getOperand(0).getType();
            assert(
                qubit_operand_mapping.find(outputQubit.getAsOpaquePointer()) !=
                qubit_operand_mapping.end());
            mlir::Value inputOperand =
                qubit_operand_mapping[outputQubit.getAsOpaquePointer()];
            auto new_inst =
                rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                    originalOp.getLoc(), llvm::makeArrayRef({qubit_type}),
                    invGateName, llvm::makeArrayRef({inputOperand}),
                    llvm::None);
            mlir::Value newResultQubit = new_inst.getResult(0);
            qubit_operand_mapping[inputQubit.getAsOpaquePointer()] =
                newResultQubit;
            output_qubit_operand_mapping[outputQubit.getAsOpaquePointer()] =
                newResultQubit;
          } else {
            mlir::Value inputQubit1 = originalOp.getOperand(0);
            mlir::Value outputQubit1 = originalOp.getResult(0);
            mlir::Value inputQubit2 = originalOp.getOperand(1);
            mlir::Value outputQubit2 = originalOp.getResult(1);
            mlir::Type qubit_type = originalOp.getOperand(0).getType();
            assert(
                qubit_operand_mapping.find(outputQubit1.getAsOpaquePointer()) !=
                qubit_operand_mapping.end());
            assert(
                qubit_operand_mapping.find(outputQubit2.getAsOpaquePointer()) !=
                qubit_operand_mapping.end());
            mlir::Value inputOperand1 =
                qubit_operand_mapping[outputQubit1.getAsOpaquePointer()];
            mlir::Value inputOperand2 =
                qubit_operand_mapping[outputQubit2.getAsOpaquePointer()];
            auto new_inst =
                rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                    originalOp.getLoc(),
                    llvm::makeArrayRef({qubit_type, qubit_type}), invGateName,
                    llvm::makeArrayRef({inputOperand1, inputOperand2}),
                    llvm::None);
            mlir::Value newResultQubit1 = new_inst.getResult(0);
            mlir::Value newResultQubit2 = new_inst.getResult(1);
            qubit_operand_mapping[inputQubit1.getAsOpaquePointer()] =
                newResultQubit1;
            qubit_operand_mapping[inputQubit2.getAsOpaquePointer()] =
                newResultQubit2;
            output_qubit_operand_mapping[outputQubit1.getAsOpaquePointer()] =
                newResultQubit1;
            output_qubit_operand_mapping[outputQubit2.getAsOpaquePointer()] =
                newResultQubit2;
          }
        } else {
          // Currently, all the gates we have here have 1 single parameter:
          assert(originalOp.params().size() == 1);
          mlir::Value angle = originalOp.params()[0];
          assert(angle.getType().isF64());
          mlir::Value float_minus_one = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -1.0));
          mlir::Value minus_angle = rewriter.create<mlir::MulFOp>(
              op.getLoc(), angle, float_minus_one);
          if (nQubits == 1) {
            mlir::Value inputQubit = originalOp.getOperand(0);
            mlir::Value outputQubit = originalOp.getResult(0);
            mlir::Type qubit_type = originalOp.getOperand(0).getType();
            assert(
                qubit_operand_mapping.find(outputQubit.getAsOpaquePointer()) !=
                qubit_operand_mapping.end());
            mlir::Value inputOperand =
                qubit_operand_mapping[outputQubit.getAsOpaquePointer()];
            auto new_inst =
                rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                    originalOp.getLoc(), llvm::makeArrayRef({qubit_type}),
                    invGateName, llvm::makeArrayRef({inputOperand}),
                    llvm::makeArrayRef({minus_angle}));
            mlir::Value newResultQubit = new_inst.getResult(0);
            qubit_operand_mapping[inputQubit.getAsOpaquePointer()] =
                newResultQubit;
            output_qubit_operand_mapping[outputQubit.getAsOpaquePointer()] =
                newResultQubit;
          } else {
            mlir::Value inputQubit1 = originalOp.getOperand(0);
            mlir::Value outputQubit1 = originalOp.getResult(0);
            mlir::Value inputQubit2 = originalOp.getOperand(1);
            mlir::Value outputQubit2 = originalOp.getResult(1);
            mlir::Type qubit_type = originalOp.getOperand(0).getType();
            assert(
                qubit_operand_mapping.find(outputQubit1.getAsOpaquePointer()) !=
                qubit_operand_mapping.end());
            assert(
                qubit_operand_mapping.find(outputQubit2.getAsOpaquePointer()) !=
                qubit_operand_mapping.end());
            mlir::Value inputOperand1 =
                qubit_operand_mapping[outputQubit1.getAsOpaquePointer()];
            mlir::Value inputOperand2 =
                qubit_operand_mapping[outputQubit2.getAsOpaquePointer()];
            auto new_inst =
                rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                    originalOp.getLoc(),
                    llvm::makeArrayRef({qubit_type, qubit_type}), invGateName,
                    llvm::makeArrayRef({inputOperand1, inputOperand2}),
                    llvm::makeArrayRef({minus_angle}));
            mlir::Value newResultQubit1 = new_inst.getResult(0);
            mlir::Value newResultQubit2 = new_inst.getResult(1);
            qubit_operand_mapping[inputQubit1.getAsOpaquePointer()] =
                newResultQubit1;
            qubit_operand_mapping[inputQubit2.getAsOpaquePointer()] =
                newResultQubit2;
            output_qubit_operand_mapping[outputQubit1.getAsOpaquePointer()] =
                newResultQubit1;
            output_qubit_operand_mapping[outputQubit2.getAsOpaquePointer()] =
                newResultQubit2;
          }
        }
      };

      std::reverse(opsToReverse.begin(), opsToReverse.end());
      for (auto &subOp : opsToReverse) {
        createInvGate(subOp);
      }
      assert(modifierEndOp);
      auto terminator = mlir::cast<mlir::quantum::ModifierEndOp>(modifierEndOp);
      for (size_t i = 0; i < terminator.qubits().size(); ++i) {
        auto iter = output_qubit_operand_mapping.find(
            terminator.qubits()[i].getAsOpaquePointer());
        assert(iter != output_qubit_operand_mapping.end());
        op.result()[i].replaceAllUsesWith(iter->second);
      }
      op.body().getBlocks().clear();
      deadOps.emplace_back(op.getOperation());
    }
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();
}

void ModifierBlockInlinerPass::runOnOperation() {
  handlePowU();
  handleCtrlU();
  handleAdjU();
}
} // namespace qcor
