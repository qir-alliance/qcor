#include "qasm3_visitor.hpp"

namespace qcor {

antlrcpp::Any qasm3_visitor::visitQuantumGateCall(
    qasm3Parser::QuantumGateCallContext* context) {
  //           quantumGateCall
  //     : quantumGateName ( LPAREN expressionList? RPAREN )?
  //     indexIdentifierList
  //     ;

  // quantumGateName
  //     : 'CX'
  //     | 'U'
  //     | 'reset'
  //     | Identifier
  //     | quantumGateModifier quantumGateName
  //     ;
  auto location = get_location(builder, file_name, context);

  auto name = context->quantumGateName()->getText();
  if (name == "CX") {
    name = "cnot";
  }
  if (name == "U") {
    name = "u3";
  }

  std::vector<mlir::Value> qbit_values, param_values;

  if (auto expression_list = context->expressionList()) {
    // we have parameters

    for (auto expression : expression_list->expression()) {
      // add parameter values:
      auto value = std::stod(expression->getText());
      auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
      mlir::Value val = builder.create<mlir::ConstantOp>(location, float_attr);
      param_values.push_back(val);
    }
  }

  for (auto idx_identifier :
       context->indexIdentifierList()->indexIdentifier()) {
    if (idx_identifier->LBRACKET()) {
      // this is a qubit indexed from an array
      auto qbit_var_name = idx_identifier->Identifier()->getText();
      auto idx_str = idx_identifier->expressionList()
                         ->expression(0)
                         ->expressionTerminator()
                         ->getText();
      auto qbit =
          get_or_extract_qubit(qbit_var_name, std::stoi(idx_str), location);
      qbit_values.push_back(qbit);
    } else {
      // this is a qubit
      auto qbit_var_name = idx_identifier->Identifier()->getText();
      auto qbit = symbol_table.get_symbol(qbit_var_name);
      qbit_values.push_back(qbit);
    }
  }
  auto str_attr = builder.getStringAttr(name);
  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qbit_values), llvm::makeArrayRef(param_values));
  return 0;
}

antlrcpp::Any qasm3_visitor::visitSubroutineCall(
    qasm3Parser::SubroutineCallContext* context) {
  // subroutineCall
  // : Identifier ( LPAREN expressionList? RPAREN )? expressionList
  // ;
  auto location = get_location(builder, file_name, context);

  auto name = context->Identifier()->getText();
  if (name == "cx") {
    name = "cnot";
  }

  if (symbol_table.has_seen_function(name)) {
      // this is a custom function
      std::cout << "HELLO THIS IS A CUSTOM FUNCTION SUBROUTINE CALL\n";
      std::cout << context->getText() << "\n";
  }
  std::vector<mlir::Value> qbit_values, param_values;

  auto qubit_expr_list_idx = 0;
  auto expression_list = context->expressionList();
  if (expression_list.size() > 1) {
    // we have parameters
    qubit_expr_list_idx = 1;

    for (auto expression : expression_list[0]->expression()) {
      // add parameter values:
      auto value = std::stod(expression->getText());
      auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
      mlir::Value val = builder.create<mlir::ConstantOp>(location, float_attr);
      param_values.push_back(val);
    }
  }

  for (auto expression : expression_list[qubit_expr_list_idx]->expression()) {
    if (expression->LBRACKET()) {
      // this is a qubit indexed from an array
      auto qbit_var_name =
          expression->expression(0)->expressionTerminator()->getText();
      auto idx_str =
          expression->expression(1)->expressionTerminator()->getText();
      auto qbit =
          get_or_extract_qubit(qbit_var_name, std::stoi(idx_str), location);
      qbit_values.push_back(qbit);
    } else {
      // this is a qubit
      auto qbit_var_name =
          expression->expressionTerminator()->Identifier()->getText();
      auto qbit = symbol_table.get_symbol(qbit_var_name);
      qbit_values.push_back(qbit);
    }
  }
  auto str_attr = builder.getStringAttr(name);
  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qbit_values), llvm::makeArrayRef(param_values));
  return 0;
}
}  // namespace qcor