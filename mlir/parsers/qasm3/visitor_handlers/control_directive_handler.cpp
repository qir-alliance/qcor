/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"
#include "mlir/Dialect/SCF/SCF.h"

namespace qcor {
void qasm3_visitor::insertLoopBreak(mlir::Location &location,
                                    mlir::OpBuilder *optional_builder) {
  mlir::OpBuilder &opBuilder = optional_builder ? *optional_builder : builder;
  mlir::Region *region = opBuilder.getInsertionBlock()->getParent();
  auto parent_op = region->getParentOp();
  mlir::scf::IfOp parentIfOp =
      mlir::dyn_cast_or_null<mlir::scf::IfOp>(parent_op);
  if (!parentIfOp) {
    // We can handle this as well, but it's really a programmers' bug.
    // Hence, just let them know.
    printErrorMessage("Illegal break directive: unconditional break.");
  }

  // Set an attribute so that we can detect this after handling this.
  parentIfOp->setAttr("control-directive",
                      mlir::IntegerAttr::get(opBuilder.getIntegerType(1), 1));
  assert(!loop_control_directive_bool_vars.empty());
  auto [cond1, cond2] = loop_control_directive_bool_vars.top();

  // Store false to both the break and continue:
  // i.e., bypass the whole for loop and the rest of the loop body:
  // (1) This bool will bypass the whole for loop body:
  opBuilder.create<mlir::StoreOp>(
      location,
      get_or_create_constant_integer_value(0, location, opBuilder.getI1Type(),
                                           symbol_table, opBuilder),
      cond1);
  // (2) This bool will bypass rest of the loop body (after this point)
  opBuilder.create<mlir::StoreOp>(
      location,
      get_or_create_constant_integer_value(0, location, opBuilder.getI1Type(),
                                           symbol_table, opBuilder),
      cond2);
}

void qasm3_visitor::insertLoopContinue(mlir::Location &location,
                                       mlir::OpBuilder *optional_builder) {
  mlir::OpBuilder &opBuilder = optional_builder ? *optional_builder : builder;
  assert(!loop_control_directive_bool_vars.empty());
  auto &[cond1, cond2] = loop_control_directive_bool_vars.top();
  // Wrap/Outline the loop body in an IfOp:
  auto continuationIfOp = opBuilder.create<mlir::scf::IfOp>(
      location, mlir::TypeRange(),
      opBuilder.create<mlir::LoadOp>(location, cond2), false);
  auto continuationThenBodyBuilder = continuationIfOp.getThenBodyBuilder();
  opBuilder = continuationThenBodyBuilder;
}

void qasm3_visitor::conditionalReturn(mlir::Location &location,
                                      mlir::Value cond, mlir::Value returnVal,
                                      mlir::OpBuilder *optional_builder) {
  mlir::OpBuilder &opBuilder = optional_builder ? *optional_builder : builder;
  assert(cond.getType() == opBuilder.getI1Type());

  auto savept = opBuilder.saveInsertionPoint();
  auto currRegion = opBuilder.getBlock()->getParent();
  assert(currRegion->getParentOp() &&
         mlir::dyn_cast_or_null<mlir::FuncOp>(currRegion->getParentOp()));

  // Create a CFG branch:
  auto thenBlock = opBuilder.createBlock(currRegion, currRegion->end());
  auto exitBlock = opBuilder.createBlock(currRegion, currRegion->end());
  opBuilder.setInsertionPointToStart(thenBlock);
  opBuilder.create<mlir::ReturnOp>(location,
                                   llvm::ArrayRef<mlir::Value>{returnVal});
  // builder.create<mlir::BranchOp>(location, exitBlock);
  opBuilder.restoreInsertionPoint(savept);
  opBuilder.create<mlir::CondBranchOp>(location, cond, thenBlock, exitBlock);
  opBuilder.setInsertionPointToStart(exitBlock);
  symbol_table.set_last_created_block(exitBlock);
}

antlrcpp::Any qasm3_visitor::visitControlDirective(
    qasm3Parser::ControlDirectiveContext *context) {
  auto location = get_location(builder, file_name, context);

  auto stmt = context->getText();

  // Strategy:
  // Converting break/continue directives to region-based control-flow (in the
  // Affine/SCF dialects) Following the direction here:
  // https://llvm.discourse.group/t/dynamic-control-flow-break-like-operation/2495/16
  // e.g., predicating every statement potentially executed after at least one
  // break on the absence of break. This doesn’t break our SCF/Affine analyses
  // and transformations, that rely on there being single block and static
  // control flow.
  // For example, with a for loop: we need to wrap the whole body in a break check
  // **and** each subsequent block after the *break/continue* point
  if (stmt == "break") {
    insertLoopBreak(location);
  } else if (stmt == "continue") {
    mlir::Region *region = builder.getInsertionBlock()->getParent();
    auto parent_op = region->getParentOp();
    mlir::scf::IfOp parentIfOp =
        mlir::dyn_cast_or_null<mlir::scf::IfOp>(parent_op);
    if (!parentIfOp) {
      // We can handle this as well, but it's really a programmers' bug.
      // Hence, just let them know.
      printErrorMessage("Illegal break directive: unconditional break.");
    }

    // Set an attribute so that we can detect this after handling this.
    parentIfOp->setAttr("control-directive",
                        mlir::IntegerAttr::get(builder.getIntegerType(1), 1));
    assert(!loop_control_directive_bool_vars.empty());
    auto [cond1, cond2] = loop_control_directive_bool_vars.top();

    // Just bypass rest of the loop body (after this point)
    // i.e., not disable the whol loop.
    builder.create<mlir::StoreOp>(
        location,
        get_or_create_constant_integer_value(0, location, builder.getI1Type(),
                                             symbol_table, builder),
        cond2);
  } else {
    printErrorMessage("we do not yet support the " + stmt +
                      " control directive.");
  }

  return 0;
}
} // namespace qcor