#include "qasm3_visitor.hpp"
#include "mlir/Dialect/SCF/SCF.h"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitQcor_test_statement(
    qasm3Parser::Qcor_test_statementContext *context) {
  auto location = get_location(builder, file_name, context);
  auto boolean_expr = context->booleanExpression();
  qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
  exp_generator.visit(boolean_expr);
  auto expr_value = exp_generator.current_value;
  // So we have a conditional result, want
  // to negate it and see if == true
  expr_value = builder.create<mlir::CmpIOp>(
      location, mlir::CmpIPredicate::ne, expr_value,
      get_or_create_constant_integer_value(1, location, builder.getI1Type(),
                                           symbol_table, builder));
  // False (not equal true is true): print message then return 1;
  auto scfIfOp = builder.create<mlir::scf::IfOp>(location, mlir::TypeRange(),
                                                 expr_value, false);
  auto thenBodyBuilder = scfIfOp.getThenBodyBuilder();
  auto sl = "QCOR Test Failure: " + context->getText() + "\n";
  llvm::StringRef string_type_name("StringType");
  mlir::Identifier dialect =
      mlir::Identifier::get("quantum", thenBodyBuilder.getContext());
  auto str_type = mlir::OpaqueType::get(thenBodyBuilder.getContext(), dialect,
                                        string_type_name);
  auto str_attr = thenBodyBuilder.getStringAttr(sl);
  std::hash<std::string> hasher;
  auto hash = hasher(sl);
  std::stringstream ss;
  ss << "__internal_string_literal__" << hash;
  std::string var_name = ss.str();
  auto var_name_attr = thenBodyBuilder.getStringAttr(var_name);
  auto string_literal =
      thenBodyBuilder.create<mlir::quantum::CreateStringLiteralOp>(
          location, str_type, str_attr, var_name_attr);
  thenBodyBuilder.create<mlir::quantum::PrintOp>(
      location, llvm::makeArrayRef(std::vector<mlir::Value>{string_literal}));

  if (region_early_return_vars.has_value()) {
    insertLoopBreak(location, &thenBodyBuilder);
    // This is in an affine region (loops)
    auto &[boolVar, returnVar] = region_early_return_vars.value();
    builder.create<mlir::StoreOp>(location, expr_value, boolVar);
    mlir::Value one_i32 = get_or_create_constant_integer_value(
        1, location, thenBodyBuilder.getI32Type(), symbol_table,
        thenBodyBuilder);
    builder.create<mlir::StoreOp>(location, one_i32, returnVar.value());
    auto &[cond1, cond2] = loop_control_directive_bool_vars.top();
    // Wrap/Outline the loop body in an IfOp:
    auto continuationIfOp = builder.create<mlir::scf::IfOp>(
        location, mlir::TypeRange(),
        builder.create<mlir::LoadOp>(location, cond2), false);
    auto continuationThenBodyBuilder = continuationIfOp.getThenBodyBuilder();
    builder = continuationThenBodyBuilder;
  } else {
    // Outside scope: just do early return
    conditionalReturn(
        location, expr_value,
        builder.create<mlir::ConstantOp>(
            location, mlir::IntegerAttr::get(thenBodyBuilder.getI32Type(), 1)));
  }

  return 0;
}
} // namespace qcor