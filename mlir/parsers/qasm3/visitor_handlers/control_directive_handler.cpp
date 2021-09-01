#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"
#include "mlir/Dialect/SCF/SCF.h"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitControlDirective(
    qasm3Parser::ControlDirectiveContext* context) {
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
    // builder.create<mlir::BranchOp>(location, current_loop_exit_block);

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
    assert(!for_loop_control_vars.empty());
    auto [cond1, cond2] = for_loop_control_vars.top();

    // Store false to both the break and continue:
    // i.e., bypass the whole for loop and the rest of the loop body:
    builder.create<mlir::StoreOp>(
        location,
        get_or_create_constant_integer_value(0, location, builder.getI1Type(),
                                             symbol_table, builder),
        cond1);
    builder.create<mlir::StoreOp>(
        location,
        get_or_create_constant_integer_value(0, location, builder.getI1Type(),
                                             symbol_table, builder),
        cond2);
  } else if (stmt == "continue") {
    // TODO: Handle this case.
    if (current_loop_incrementor_block) {
      builder.create<mlir::BranchOp>(location, current_loop_incrementor_block);
    } else if (current_loop_header_block) {
      // this is a while loop
      builder.create<mlir::BranchOp>(location, current_loop_header_block);
    } else {
      printErrorMessage(
          "Something went wrong with continue, no valid block to branch to.");
    }
  } else {
    printErrorMessage("we do not yet support the " + stmt +
                      " control directive.");
  }

  return 0;
}
}  // namespace qcor