#include "qasm3_visitor.hpp"

namespace qcor {

void qasm3_visitor::createInstOps_HandleBroadcast(
    std::string name, std::vector<mlir::Value> qbit_values,
    std::vector<mlir::Value> param_values, mlir::Location location,
    antlr4::ParserRuleContext* context) {
  auto has_array_type = [this](auto& value_vector) {
    for (auto v : value_vector) {
      if (v.getType() == array_type) {
        return true;
      }
    }
    return false;
  };
  auto str_attr = builder.getStringAttr(name);

  // FIXME Extremely hacky way to handle gate broadcasting
  // The cases we consider are...
  // QINST qubit
  // QINST qubit[] -> QINST qubit[j] for all j
  // QINST qubit qubit
  // QINST qubit qubit[] -> QINST qubit qubit[j] for all j
  // QINST qubit[] qubit -> QINST qubit[j] qubit for all j
  // QINST qubit[] qubit[] -> QINST qubit[j] qubit[j] for all j
  if (has_array_type(qbit_values)) {
    if (qbit_values.size() == 1) {
      auto n = qbit_values[0]
                   .getDefiningOp<mlir::quantum::QallocOp>()
                   .size()
                   .getLimitedValue();
      for (int i = 0; i < n; i++) {
        auto qubit_type = get_custom_opaque_type("Qubit", builder.getContext());

        auto extract_value = builder.create<mlir::quantum::ExtractQubitOp>(
            location, qubit_type, qbit_values[0],
            get_or_create_constant_integer_value(
                i, location, builder.getI64Type(), symbol_table, builder));
        builder.create<mlir::quantum::InstOp>(
            location, mlir::NoneType::get(builder.getContext()), str_attr,
            llvm::makeArrayRef(std::vector<mlir::Value>{extract_value}),
            llvm::makeArrayRef(param_values));
      }
    } else if (qbit_values.size() == 2) {
      if (qbit_values[0].getType() == array_type &&
          qbit_values[1].getType() == array_type) {
        auto n = qbit_values[0]
                     .getDefiningOp<mlir::quantum::QallocOp>()
                     .size()
                     .getLimitedValue();
        auto m = qbit_values[0]
                     .getDefiningOp<mlir::quantum::QallocOp>()
                     .size()
                     .getLimitedValue();
        if (n != m) {
          printErrorMessage("Gate broadcast must be on registers of same size.",
                            context);
        }

        for (int i = 0; i < n; i++) {
          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());

          auto extract_value_n = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qbit_values[0],
              get_or_create_constant_integer_value(
                  i, location, builder.getI64Type(), symbol_table, builder));
          auto extract_value_m = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qbit_values[1],
              get_or_create_constant_integer_value(
                  i, location, builder.getI64Type(), symbol_table, builder));

          builder.create<mlir::quantum::InstOp>(
              location, mlir::NoneType::get(builder.getContext()), str_attr,
              llvm::makeArrayRef(
                  std::vector<mlir::Value>{extract_value_n, extract_value_m}),
              llvm::makeArrayRef(param_values));
        }

      } else if (qbit_values[0].getType() == array_type &&
                 qbit_values[1].getType() != array_type) {
        auto n = qbit_values[0]
                     .getDefiningOp<mlir::quantum::QallocOp>()
                     .size()
                     .getLimitedValue();
        for (int i = 0; i < n; i++) {
          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());

          auto extract_value = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qbit_values[0],
              get_or_create_constant_integer_value(
                  i, location, builder.getI64Type(), symbol_table, builder));

          builder.create<mlir::quantum::InstOp>(
              location, mlir::NoneType::get(builder.getContext()), str_attr,
              llvm::makeArrayRef(
                  std::vector<mlir::Value>{extract_value, qbit_values[1]}),
              llvm::makeArrayRef(param_values));
        }
      } else if (qbit_values[0].getType() != array_type &&
                 qbit_values[1].getType() == array_type) {
        auto n = qbit_values[1]
                     .getDefiningOp<mlir::quantum::QallocOp>()
                     .size()
                     .getLimitedValue();
        for (int i = 0; i < n; i++) {
          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());

          auto extract_value = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qbit_values[1],
              get_or_create_constant_integer_value(
                  i, location, builder.getI64Type(), symbol_table, builder));

          builder.create<mlir::quantum::InstOp>(
              location, mlir::NoneType::get(builder.getContext()), str_attr,
              llvm::makeArrayRef(
                  std::vector<mlir::Value>{qbit_values[0], extract_value}),
              llvm::makeArrayRef(param_values));
        }
      }
    } else {
      printErrorMessage(
          "can only broadcast gates with one or two qubit registers");
    }
  } else {
    builder.create<mlir::quantum::InstOp>(
        location, mlir::NoneType::get(builder.getContext()), str_attr,
        llvm::makeArrayRef(qbit_values), llvm::makeArrayRef(param_values));
  }
}

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
      mlir::Value val;
      try {
        auto value = std::stod(expression->getText());
        auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
        val = builder.create<mlir::ConstantOp>(location, float_attr);
      } catch (...) {
        qasm3_expression_generator exp_generator(builder, symbol_table,
                                                 file_name);
        exp_generator.visit(expression);
        val = exp_generator.current_value;
      }
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
      mlir::Value value;
      try {
        value = get_or_extract_qubit(qbit_var_name, std::stoi(idx_str),
                                     location, symbol_table, builder);
      } catch (...) {
        if (symbol_table.has_symbol(idx_str)) {
          auto qubits = symbol_table.get_symbol(qbit_var_name);
          auto qbit = symbol_table.get_symbol(idx_str);

          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());

          value = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qubits, qbit);
        } else {
          printErrorMessage(
              "Invalid measurement index on the given qubit register: " +
              qbit_var_name + ", " + idx_str);
        }
      }

      // auto qbit =
      //     get_or_extract_qubit(qbit_var_name, std::stoi(idx_str), location);
      qbit_values.push_back(value);
    } else {
      // this is a qubit
      auto qbit_var_name = idx_identifier->Identifier()->getText();
      auto qbit = symbol_table.get_symbol(qbit_var_name);
      qbit_values.push_back(qbit);
    }
  }

  createInstOps_HandleBroadcast(name, qbit_values, param_values, location,
                                context);

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
      if (arg.getType().isa<mlir::MemRefType>()) {
        arg = builder.create<mlir::LoadOp>(
            location, arg,
            get_or_create_constant_index_value(0, location, 64, symbol_table,
                                               builder));
      }
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

  std::vector<mlir::Value> qbit_values, param_values;

  auto qubit_expr_list_idx = 0;
  auto expression_list = context->expressionList();
  if (expression_list.size() > 1) {
    // we have parameters
    qubit_expr_list_idx = 1;

    for (auto expression : expression_list[0]->expression()) {
      // add parameter values:
      qasm3_expression_generator qubit_exp_generator(builder, symbol_table,
                                                     file_name);
      qubit_exp_generator.visit(expression);
      param_values.push_back(qubit_exp_generator.current_value);
    }
  }

  auto str_attr = builder.getStringAttr(name);

  auto n_qubit_args = expression_list[qubit_expr_list_idx]->expression().size();
  for (auto expression : expression_list[qubit_expr_list_idx]->expression()) {
    qasm3_expression_generator qubit_exp_generator(builder, symbol_table,
                                                   file_name, qubit_type);
    qubit_exp_generator.visit(expression);
    auto qbit_or_qreg = qubit_exp_generator.current_value;
    qbit_values.push_back(qubit_exp_generator.current_value);
  }

  createInstOps_HandleBroadcast(name, qbit_values, param_values, location,
                                context);
  return 0;
}
}  // namespace qcor