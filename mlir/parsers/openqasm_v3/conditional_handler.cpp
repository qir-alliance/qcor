
#include "qasm3_visitor.hpp"
#include "expression_handler.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitBranchingStatement(
    qasm3Parser::BranchingStatementContext* context) {
  auto location = get_location(builder, file_name, context);

  // Get the conditional expression
  auto conditional_expr = context->expression();

  qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
  exp_generator.visit(conditional_expr);
  auto expr_value = exp_generator.current_value;

  // build up the program block
  auto currRegion = builder.getBlock()->getParent();
  auto savept = builder.saveInsertionPoint();
  auto thenBlock = builder.createBlock(currRegion, currRegion->end());
  auto elseBlock = builder.createBlock(currRegion, currRegion->end());

  // Build up the THEN Block, add return at end
  builder.setInsertionPointToStart(thenBlock);
  symbol_table.enter_new_scope();
  // Get the conditional code and visit the nodes
  auto conditional_code = context->programBlock(0);
  visitChildren(conditional_code);
  symbol_table.exit_scope();
  // Add a return from the block
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                 llvm::ArrayRef<mlir::Value>());

  // If we have a second program block then we have an else stmt
  builder.setInsertionPointToStart(elseBlock);
  if (context->programBlock().size() == 2) {
    std::cout << "WE HAVE AN ELSE BLOCK\n";
    symbol_table.enter_new_scope();
    visitChildren(context->programBlock(1));
    symbol_table.exit_scope();
  }
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                 llvm::ArrayRef<mlir::Value>());

  // Restore the insertion point and create the conditional statement
  builder.restoreInsertionPoint(savept);
  builder.create<mlir::CondBranchOp>(location, expr_value, thenBlock,
                                     elseBlock);

  return 0;  // visitChildren(context);
}
}  // namespace qcor