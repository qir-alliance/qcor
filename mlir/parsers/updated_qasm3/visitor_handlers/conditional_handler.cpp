
#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitBranchingStatement(
    qasm3Parser::BranchingStatementContext* context) {
  auto location = get_location(builder, file_name, context);

  // Get the conditional expression
  auto conditional_expr = context->booleanExpression();

  qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
  exp_generator.visit(conditional_expr);
  auto expr_value = exp_generator.current_value;

  // build up the program block
  auto currRegion = builder.getBlock()->getParent();
  auto savept = builder.saveInsertionPoint();
  auto thenBlock = builder.createBlock(currRegion, currRegion->end());
  auto elseBlock = builder.createBlock(currRegion, currRegion->end());
  mlir::Block* exitBlock = nullptr;
  if (context->programBlock().size() == 2) {
    exitBlock = builder.createBlock(currRegion, currRegion->end());

  } else {
    exitBlock = elseBlock;
  }

  // Build up the THEN Block, add return at end
  builder.setInsertionPointToStart(thenBlock);
  symbol_table.enter_new_scope();
  // Get the conditional code and visit the nodes
  auto conditional_code = context->programBlock(0);
  visitChildren(conditional_code);
  
  // Need to check if we have a branch out of 
  // this thenBlock, if so do not add a branch 
  // to the exitblock
  mlir::Operation& last_op = thenBlock->back();
  auto branchOps = thenBlock->getOps<mlir::BranchOp>();
  auto branch_to_exit = true;
  for (auto b : branchOps) {
    if (b.dest() == current_loop_exit_block || 
        b.dest() == current_loop_header_block ||
        b.dest() == current_loop_incrementor_block) {
      branch_to_exit = false;
      break;
    }
  }

  if (branch_to_exit) {
    builder.create<mlir::BranchOp>(location, exitBlock);
  }
  symbol_table.exit_scope();

  // If we have a second program block then we have an else stmt
  builder.setInsertionPointToStart(elseBlock);
  if (context->programBlock().size() == 2) {
    symbol_table.enter_new_scope();
    visitChildren(context->programBlock(1));
    builder.create<mlir::BranchOp>(location, exitBlock);
    symbol_table.exit_scope();
  }

  // Restore the insertion point and create the conditional statement
  builder.restoreInsertionPoint(savept);
  builder.create<mlir::CondBranchOp>(location, expr_value, thenBlock,
                                     elseBlock);
  builder.setInsertionPointToStart(exitBlock);

  symbol_table.set_last_created_block(exitBlock);

  return 0;  // visitChildren(context);
}
}  // namespace qcor