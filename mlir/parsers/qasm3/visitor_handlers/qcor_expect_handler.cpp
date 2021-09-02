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
  auto integer_attr = mlir::IntegerAttr::get(thenBodyBuilder.getI32Type(), 1);
  auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
  thenBodyBuilder.create<mlir::ReturnOp>(location, llvm::ArrayRef<mlir::Value>(ret));

  return 0;
}
} // namespace qcor