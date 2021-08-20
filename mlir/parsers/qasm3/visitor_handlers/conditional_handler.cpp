
#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace {
// ATM, we don't try to convert everything to the
// special Quantum If-Then-Else Op.
// Rather, we aim for very narrow use-case that we're sure
// that this can be done.
// i.e., this serve mainly as a stop-gap before fully-FTQC runtimes become
// available.

// FIXME: Define a Target Capability setting and make the compiler aware of that.

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

// Callable running-off captured vars...
mlir::Value create_capture_callable_gen(
    mlir::OpBuilder &builder, const std::string &func_name,
    mlir::ModuleOp &moduleOp, mlir::FuncOp &wrapped_func,
    std::vector<mlir::Value> &captured_vars) {
  auto context = builder.getContext();
  auto main_block = builder.saveInsertionPoint();
  mlir::Identifier dialect = mlir::Identifier::get("quantum", context);
  llvm::StringRef tuple_type_name("Tuple");
  auto tuple_type = mlir::OpaqueType::get(context, dialect, tuple_type_name);
  llvm::StringRef callable_type_name("Callable");
  auto callable_type =
      mlir::OpaqueType::get(context, dialect, callable_type_name);
  const std::vector<mlir::Type> argument_types{tuple_type, tuple_type,
                                               tuple_type};
  auto func_type = builder.getFunctionType(argument_types, llvm::None);
  const std::string BODY_WRAPPER_SUFFIX = "__body__wrapper";
  // Body wrapper:
  const std::string wrapper_fn_name = func_name + BODY_WRAPPER_SUFFIX;
  mlir::FuncOp function_op(mlir::FuncOp::create(builder.getUnknownLoc(),
                                                wrapper_fn_name, func_type));
  function_op.setVisibility(mlir::SymbolTable::Visibility::Private);
  auto &entryBlock = *function_op.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);
  auto arguments = entryBlock.getArguments();
  assert(arguments.size() == 3);
  // Unpack from **captured** vars (not input args...)
  // i.e., Tuple # 0
  mlir::Value arg_tuple = arguments[0];
  auto fn_type = wrapped_func.getType().cast<mlir::FunctionType>();
  mlir::TypeRange arg_types(fn_type.getInputs());
  auto unpackOp = builder.create<mlir::quantum::TupleUnpackOp>(
      builder.getUnknownLoc(), arg_types, arg_tuple);
  auto call_op = builder.create<mlir::CallOp>(builder.getUnknownLoc(),
                                              wrapped_func, unpackOp.result());
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
  moduleOp.push_back(function_op);
  
  // !! We only ever invoke the body functor, create dummy functors for adj/ctrl
  for (const auto &suffix :
       {"__adj__wrapper", "__ctl__wrapper", "__ctladj__wrapper"}) {
    builder.restoreInsertionPoint(main_block);
    const std::string temp_fn_name = func_name + suffix;
    mlir::FuncOp fn_op(
        mlir::FuncOp::create(builder.getUnknownLoc(), temp_fn_name, func_type));
    fn_op.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto &entryBlock = *fn_op.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    moduleOp.push_back(fn_op);
  }

  builder.restoreInsertionPoint(main_block);
  auto callable_create_op = builder.create<mlir::quantum::CreateCallableOp>(
      builder.getUnknownLoc(), callable_type,
      builder.getSymbolRefAttr(wrapped_func),
      /*captures*/ llvm::makeArrayRef(captured_vars));
  return callable_create_op;
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
  if (bit_check_conditional.has_value() &&
      context->programBlock().size() == 1 &&
      symbol_table.try_lookup_meas_result(bit_check_conditional->var_name)
          .has_value()) {
    auto meas_var =
        symbol_table.try_lookup_meas_result(bit_check_conditional->var_name);
    
    // Strategy: we wrap the body as a Callable capturing
    // all avaiable variables at the current scope.
    // Note: we could detect which variables are used in the 
    // conditional block body to be included in the capture.
    auto all_vars = symbol_table.get_all_visible_symbols();
    auto main_block = builder.saveInsertionPoint();
    std::vector<mlir::Type> argument_types;
    std::vector<std::string> argument_names;
    std::vector<mlir::Value> argument_values;
    // Narrow the list of supported types for tuple unpack...
    // We don't support all types atm.
    for (auto &[k, v] : all_vars) {
      // QIR types and Float (rotation angles)
      if (v.getType().isa<mlir::OpaqueType>() ||
          v.getType().isa<mlir::FloatType>()) {
        argument_names.emplace_back(k);
        argument_values.emplace_back(v);
        argument_types.emplace_back(v.getType());
      }
    }

    // Use the ANTLR node ptr (hex) as id for this temp. function
    const auto toString = [](auto *antr_node) {
      std::stringstream ss;
      ss << (void *)antr_node;
      return ss.str();
    };
    const std::string tmp_func_name =
        "if_body_" + toString(context->programBlock(0));
    auto func_type = builder.getFunctionType(argument_types, llvm::None);
    auto proto =
        mlir::FuncOp::create(builder.getUnknownLoc(), tmp_func_name, func_type);
    mlir::FuncOp function(proto);
    function.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    symbol_table.enter_new_scope();
    auto arguments = entryBlock.getArguments();
    for (int i = 0; i < arguments.size(); i++) {
      symbol_table.add_symbol(argument_names[i], arguments[i], {}, true);
    }
    visitChildren(context->programBlock(0));
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    builder.restoreInsertionPoint(main_block);
    symbol_table.exit_scope();
    symbol_table.add_seen_function(tmp_func_name, function);
    symbol_table.set_last_created_block(nullptr);
    for (int i = 0; i < arguments.size(); ++i) {
      symbol_table.replace_symbol(symbol_table.get_symbol(argument_names[i]),
                                  argument_values[i]);
    }
    m_module.push_back(function);

    auto then_body_callable = create_capture_callable_gen(
        builder, tmp_func_name, m_module, function, argument_values);
    auto ifOp = builder.create<mlir::quantum::ConditionalOp>(
        location, meas_var.value(), then_body_callable);
    // Done
    return 0;
  }

  // TODO: The below code could be rewritten to an AffineIfOp/SCF::IfOp:
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