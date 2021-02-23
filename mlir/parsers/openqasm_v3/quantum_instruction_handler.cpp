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

antlrcpp::Any qasm3_visitor::visitKernelCall(
    qasm3Parser::KernelCallContext* context) {
  auto location = get_location(builder, file_name, context);

  if (context->Identifier()->getText() == "print") {
    std::vector<mlir::Value> print_args;
    auto expr_list = context->expressionList();
    for (auto exp : expr_list->expression()) {
      // auto exp_str = exp->getText();
      qasm3_expression_generator exp_generator(builder, symbol_table,
                                               file_name);
      exp_generator.visit(exp);

      auto arg = exp_generator.current_value;
      print_args.push_back(arg);
    }
    builder.create<mlir::quantum::PrintOp>(location,
                                           llvm::makeArrayRef(print_args));
    return 0;
  }

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

  if (name == "print") {
    auto one_expr_list = context->expressionList()[0];
    std::vector<mlir::Value> print_args;
    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(one_expr_list->expression()[0]);
    auto arg = exp_generator.current_value;

    print_args.push_back(arg);
    builder.create<mlir::quantum::PrintOp>(location,
                                           llvm::makeArrayRef(print_args));
    return 0;
  }

  if (symbol_table.has_seen_function(name)) {
    // this is a custom function
    std::cout << "HELLO THIS IS A CUSTOM FUNCTION SUBROUTINE CALL\n";
    std::cout << context->getText() << "\n";
    printErrorMessage("Throwing this here to stop us.");
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

      auto qreg = symbol_table.get_symbol(qbit_var_name);

      auto qalloc_op = qreg.getDefiningOp<mlir::quantum::QallocOp>();
      // auto qubits = qalloc_op.qubits();

      mlir::Value qbit;
      if (symbol_table.has_symbol(idx_str)) {
        // This is a variable qubit register index, see if its in the symbol
        // table
        auto idx = symbol_table.get_symbol(idx_str);
        if (idx.getType().isa<mlir::MemRefType>() &&
            idx.getType()
                .cast<mlir::MemRefType>()
                .getElementType()
                .isa<mlir::IntegerType>() &&
            idx.getType().cast<mlir::MemRefType>().getRank() == 1) {
          // If this is a memref for an integer of rank 1,
          // then it can be used as a qubit register index.
          llvm::ArrayRef<mlir::Value> zero_index(
              get_or_create_constant_index_value(0, location));
          idx = builder.create<mlir::LoadOp>(location, idx, zero_index);
        }
        qbit = builder.create<mlir::quantum::ExtractQubitOp>(
            location, qubit_type, qreg, idx);
      } else {
        if (qalloc_op) {
          auto qreg_size = qalloc_op.size().getLimitedValue();

          if (std::stoi(idx_str) >= qreg_size) {
            printErrorMessage("Invalid qubit index: " + idx_str +
                              " >= qubit register size " +
                              std::to_string(qreg_size));
          }
        }
        qbit =
            get_or_extract_qubit(qbit_var_name, std::stoi(idx_str), location);
      }
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