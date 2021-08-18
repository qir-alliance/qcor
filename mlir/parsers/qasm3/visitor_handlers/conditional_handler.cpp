
#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace {
// ATM, we don't try to convert everything to the
// special Quantum If-Then-Else Op.
// Rather, we aim for very narrow use-case that we're sure
// that this can be done.
// i.e., this serve mainly as a stop-gap before fully-FTQC runtimes become
// available.

// Capture binary comparison conditional.
// Note: currently, only bit-level is modeled.
// (Some backends, e.g., Honeywell, support a richer set of comparison ops).
struct BitComparisonExpression {
  std::string var_name;
  bool comparison_value;
  BitComparisonExpression(const std::string &var, bool val)
      : var_name(var), comparison_value(val) {}
};

// Test if this is a simple bit check conditional:
// e.g.,
// if (b) or if (b==1), etc.
std::optional<BitComparisonExpression> tryParseSimpleBooleanExpression(
    qasm3Parser::BooleanExpressionContext &boolean_expr) {
  // std::cout << "conditional expr: " << boolean_expr.getText() << "\n";
  if (boolean_expr.comparsionExpression()) {
    auto &comp_expr = *(boolean_expr.comparsionExpression());
    if (comp_expr.expression().size() == 1) {
      return BitComparisonExpression(comp_expr.expression(0)->getText(), true);
    }
    if (comp_expr.expression().size() == 2 && comp_expr.relationalOperator()) {
      if (comp_expr.relationalOperator()->getText() == "==") {
        if (comp_expr.expression(1)->getText() == "1") {
          return BitComparisonExpression(comp_expr.expression(0)->getText(),
                                         true);
        } else {
          return BitComparisonExpression(comp_expr.expression(0)->getText(),
                                         false);
        }
      }
      if (comp_expr.relationalOperator()->getText() == "!=") {
        if (comp_expr.expression(1)->getText() == "1") {
          return BitComparisonExpression(comp_expr.expression(0)->getText(),
                                         false);
        } else {
          return BitComparisonExpression(comp_expr.expression(0)->getText(),
                                         true);
        }
      }
    }
  }

  return std::nullopt;
}
} // namespace

namespace qcor {
antlrcpp::Any qasm3_visitor::visitBranchingStatement(
    qasm3Parser::BranchingStatementContext* context) {
  auto location = get_location(builder, file_name, context);

  // Get the conditional expression
  auto conditional_expr = context->booleanExpression();

  auto bit_check_conditional =
      tryParseSimpleBooleanExpression(*conditional_expr);
  // Currently, we're only support If (not else yet)
  auto meas_var =
      symbol_table.try_lookup_meas_result(bit_check_conditional->var_name);
  if (bit_check_conditional.has_value() &&
      context->programBlock().size() == 1 && meas_var.has_value()) {
    std::cout << "This is a simple Measure check\n";
    auto nisqIfStmt = builder.create<mlir::quantum::ConditionalOp>(
        location, meas_var.value());
  }

  // Map it to a Value
  qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
  exp_generator.visit(conditional_expr);
  auto expr_value = exp_generator.current_value;

  // build up the program block
  auto currRegion = builder.getBlock()->getParent();
  auto savept = builder.saveInsertionPoint();
  auto thenBlock = builder.createBlock(currRegion, currRegion->end());
  auto elseBlock = builder.createBlock(currRegion, currRegion->end());
  mlir::Block* exitBlock = nullptr;
  // If we have an else block from programBlock,
  // then create a stand alone exit block that both
  // then and else can fall to
  if (context->programBlock().size() == 2) {
    exitBlock = builder.createBlock(currRegion, currRegion->end());
  } else {
    exitBlock = elseBlock;
  }

  // Build up the THEN Block
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
    branch_to_exit = true;
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
  }

  // Restore the insertion point and create the conditional statement
  builder.restoreInsertionPoint(savept);
  builder.create<mlir::CondBranchOp>(location, expr_value, thenBlock,
                                     elseBlock);
  builder.setInsertionPointToStart(exitBlock);

  symbol_table.set_last_created_block(exitBlock);

  return 0;
}
}  // namespace qcor