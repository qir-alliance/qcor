/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "qasm3_visitor.hpp"

namespace qcor {

std::vector<mlir::Value> qasm3_visitor::createInstOps_HandleBroadcast(
    std::string name, std::vector<mlir::Value> qbit_values,
    std::vector<std::string> qreg_names,
    std::vector<std::string> symbol_table_qbit_keys,
    std::vector<mlir::Value> param_values, mlir::Location location,
    antlr4::ParserRuleContext* context) {
  std::vector<mlir::Value> updated_qubit_values;
  
  auto has_array_type = [this](auto& value_vector) {
    for (auto v : value_vector) {
      if (v.getType() == array_type) {
        return true;
      }
    }
    return false;
  };

  auto get_qreg_size = [&, this](mlir::Value qreg_value,
                                 const std::string qreg_name) {
    uint64_t nqubits = 0;
    if (auto op = qreg_value.getDefiningOp<mlir::quantum::QallocOp>()) {
      nqubits = op.size().getLimitedValue();
    } else {
      auto attributes = symbol_table.get_variable_attributes(qreg_name);
      if (!attributes.empty()) {
        try {
          nqubits = std::stoi(attributes[0]);
        } catch (...) {
          printErrorMessage("Could not infer qubit[] size from block argument.",
                            context);
        }
      } else {
        printErrorMessage(
            "Could not infer qubit[] size from block argument. No size "
            "attribute for variable in symbol table.",
            context);
      }
    }
    return nqubits;
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
      auto n = get_qreg_size(qbit_values[0], qreg_names[0]);
      for (int i = 0; i < n; i++) {
        auto qubit_type = get_custom_opaque_type("Qubit", builder.getContext());

        auto extract_value = get_or_extract_qubit(qreg_names[0], i, location,
                                                  symbol_table, builder);

        std::vector<mlir::Type> ret_types;
        for (auto q : qbit_values) {
          ret_types.push_back(qubit_type);
        }
        auto inst = builder.create<mlir::quantum::ValueSemanticsInstOp>(
            location, llvm::makeArrayRef(ret_types), str_attr,
            llvm::makeArrayRef(extract_value),
            llvm::makeArrayRef(param_values));

        // Replace qbit_values in symbol table with new result qubits
        auto return_vals = inst.result();
        int ii = 0;
        for (auto result : return_vals) {
          symbol_table.replace_symbol(extract_value, result);
          updated_qubit_values.emplace_back(result);
          ii++;
        }
      }
    } else if (qbit_values.size() == 2) {
      if (qbit_values[0].getType() == array_type &&
          qbit_values[1].getType() == array_type) {
        auto n = get_qreg_size(qbit_values[0], qreg_names[0]);
        auto m = get_qreg_size(qbit_values[1], qreg_names[1]);

        // This case is cx qarray, rarray;

        if (n != m) {
          printErrorMessage("Gate broadcast must be on registers of same size.",
                            context);
        }

        for (int i = 0; i < n; i++) {
          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());

          auto extract_value_n = get_or_extract_qubit(
              qreg_names[0], i, location, symbol_table, builder);
          auto extract_value_m = get_or_extract_qubit(
              qreg_names[1], i, location, symbol_table, builder);

          std::vector<mlir::Type> ret_types;
          for (auto q : qbit_values) {
            ret_types.push_back(qubit_type);
          }
          auto inst = builder.create<mlir::quantum::ValueSemanticsInstOp>(
              location, llvm::makeArrayRef(ret_types), str_attr,
              llvm::makeArrayRef({extract_value_n, extract_value_m}),
              llvm::makeArrayRef(param_values));

          // Replace qbit_values in symbol table with new result qubits
          auto return_vals = inst.result();
          int ii = 0;
          for (auto result : return_vals) {
            symbol_table.replace_symbol(
                (ii == 0 ? extract_value_n : extract_value_m), result);
            updated_qubit_values.emplace_back(result);
            ii++;
          }
        }

      } else if (qbit_values[0].getType() == array_type &&
                 qbit_values[1].getType() != array_type) {
        auto n = get_qreg_size(qbit_values[0], qreg_names[0]);
        mlir::Value v = qbit_values[1];

        for (int i = 0; i < n; i++) {
          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());

          // This case is cx qarray, r;

          auto extract_value = get_or_extract_qubit(qreg_names[0], i, location,
                                                    symbol_table, builder);

          std::vector<mlir::Type> ret_types;
          for (auto q : qbit_values) {
            ret_types.push_back(qubit_type);
          }
          auto inst = builder.create<mlir::quantum::ValueSemanticsInstOp>(
              location, llvm::makeArrayRef(ret_types), str_attr,
              llvm::makeArrayRef({extract_value, v}),
              llvm::makeArrayRef(param_values));

          // Replace qbit_values in symbol table with new result qubits
          auto return_vals = inst.result();
          int ii = 0;
          for (auto result : return_vals) {
            symbol_table.replace_symbol((ii == 0 ? extract_value : v), result);
            updated_qubit_values.emplace_back(result);
            ii++;
          }
          v = return_vals[1];
        }
      } else if (qbit_values[0].getType() != array_type &&
                 qbit_values[1].getType() == array_type) {
        auto n = get_qreg_size(qbit_values[1], qreg_names[1]);
        // This is cx q, rarray

        mlir::Value v = qbit_values[0];
        for (int i = 0; i < n; i++) {
          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());

          auto extract_value = get_or_extract_qubit(qreg_names[1], i, location,
                                                    symbol_table, builder);

          std::vector<mlir::Type> ret_types;
          for (auto q : qbit_values) {
            ret_types.push_back(qubit_type);
          }
          auto inst = builder.create<mlir::quantum::ValueSemanticsInstOp>(
              location, llvm::makeArrayRef(ret_types), str_attr,
              llvm::makeArrayRef({v, extract_value}),
              llvm::makeArrayRef(param_values));

          // Replace qbit_values in symbol table with new result qubits
          auto return_vals = inst.result();
          int ii = 0;
          for (auto result : return_vals) {
            symbol_table.replace_symbol((ii == 0 ? v : extract_value), result);
            updated_qubit_values.emplace_back(result);
            ii++;
          }
          v = return_vals[0];
        }
      }
    } else {
      printErrorMessage(
          "can only broadcast gates with one or two qubit registers");
    }
  } else {
    if (symbol_table_qbit_keys.empty()) {
      builder.create<mlir::quantum::InstOp>(
          location, mlir::NoneType::get(builder.getContext()), str_attr,
          llvm::makeArrayRef(qbit_values), llvm::makeArrayRef(param_values));
    } else {
      std::vector<mlir::Type> ret_types;
      for (auto q : qbit_values) {
        ret_types.push_back(qubit_type);
      }

      auto inst = builder.create<mlir::quantum::ValueSemanticsInstOp>(
          location, llvm::makeArrayRef(ret_types), str_attr,
          llvm::makeArrayRef(qbit_values), llvm::makeArrayRef(param_values));

      // Replace qbit_values in symbol table with new result qubits
      auto return_vals = inst.result();
      int i = 0;
      for (auto result : return_vals) {
        symbol_table.replace_symbol(qbit_values[i], result);
        updated_qubit_values.emplace_back(result);
        i++;
      }
    }
  }
  return updated_qubit_values;
}

antlrcpp::Any qasm3_visitor::visitQuantumGateCall(
    qasm3Parser::QuantumGateCallContext* context) {
  // quantumGateCall
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

  std::vector<qasm3Parser::QuantumGateModifierContext*> modifiers;
  if (context->quantumGateName()->quantumGateModifier()) {
    auto qgn = context->quantumGateName();
    while (auto qgm = qgn->quantumGateModifier()) {
      modifiers.push_back(qgm);
      qgn = qgn->quantumGateName();
    }
  }

  for (auto m : modifiers) {
    auto pos = name.find(m->getText());
    name.erase(pos, m->getText().length());
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
        if (val.getType().isa<mlir::MemRefType>()) {
          if (!val.getType()
                   .cast<mlir::MemRefType>()
                   .getElementType()
                   .isIntOrFloat()) {
            printErrorMessage(
                "Variable classical parameter for quantum instruction is not a "
                "float or int.",
                context, {val});
          }

          auto shape = val.getType().cast<mlir::MemRefType>().getShape();
          if (!shape.empty() && shape[0] == 1) {
            val = builder.create<mlir::LoadOp>(
                location, val,
                get_or_create_constant_index_value(0, location, 64,
                                                   symbol_table, builder));
          }
        }
      }
      param_values.push_back(val);
    }
  }

  std::vector<std::string> qreg_names, qubit_symbol_table_keys;
  for (auto idx_identifier :
       context->indexIdentifierList()->indexIdentifier()) {
    auto qbit_var_name = idx_identifier->Identifier()->getText();
    qreg_names.push_back(qbit_var_name);
    if (idx_identifier->LBRACKET()) {
      // this is a qubit indexed from an array
      auto idx_str = idx_identifier->expressionList()->expression(0)->getText();
      const auto qubit_symbol_name =
          symbol_table.array_qubit_symbol_name(qbit_var_name, idx_str);
      mlir::Value value;
      try {
        // try catch is on this std::stoi(), if idx_str is not an integer,
        // then we drop out and try to evaluate the expression.
        const auto idx_val = std::stoi(idx_str);
        // Note: always use get_or_extract_qubit which has built-in qubit SSA
        // validation/adjust.
        value = get_or_extract_qubit(qbit_var_name, idx_val, location,
                                     symbol_table, builder);
      } catch (...) {
        if (symbol_table.has_symbol(idx_str)) {
          auto qubits = symbol_table.get_symbol(qbit_var_name);
          auto qbit = symbol_table.get_symbol(idx_str);

          auto qubit_type =
              get_custom_opaque_type("Qubit", builder.getContext());
          if (!qbit.getType().isa<mlir::IntegerType>()) {
            qbit = builder.create<mlir::IndexCastOp>(
                location, builder.getI64Type(), qbit);
          }

          // This is qubit extract by a variable index:
          symbol_table.invalidate_qubit_extracts(qbit_var_name);
          value = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qubits, qbit);
          if (!symbol_table.has_symbol(qubit_symbol_name))
            symbol_table.add_symbol(qubit_symbol_name, value);
        } else {
          qasm3_expression_generator exp_generator(builder, symbol_table,
                                                   file_name, qubit_type);
          exp_generator.visit(idx_identifier->expressionList()->expression(0));
          value = exp_generator.current_value;
          if (!value.getDefiningOp<mlir::quantum::ExtractQubitOp>()) {
            auto qubits = symbol_table.get_symbol(qbit_var_name);

            auto qubit_type =
                get_custom_opaque_type("Qubit", builder.getContext());

            if (value.getType().getIntOrFloatBitWidth() < 64) {
              value = builder.create<mlir::ZeroExtendIOp>(location, value,
                                                          builder.getI64Type());
            }

            if (!value.getType().isa<mlir::IntegerType>()) {
              value = builder.create<mlir::IndexCastOp>(
                  location, builder.getI64Type(), value);
            }

            value = builder.create<mlir::quantum::ExtractQubitOp>(
                location, qubit_type, qubits, value);
            if (!symbol_table.has_symbol(qubit_symbol_name))
              symbol_table.add_symbol(qubit_symbol_name, value);
          }
        }
      }

      qbit_values.push_back(value);
      qubit_symbol_table_keys.push_back(qubit_symbol_name);

    } else {
      // this is a qubit
      auto qbit = symbol_table.get_symbol(qbit_var_name);
      qbit_values.push_back(qbit);
      qubit_symbol_table_keys.push_back(qbit_var_name);
    }
  }

  std::stack<std::pair<mlir::OpBuilder::InsertPoint, mlir::Operation *>>
      modifier_insertion_points_stack;
  for (auto m : modifiers) {
    if (m->getText().find("pow") != std::string::npos) {
      mlir::Value power;
      if (symbol_table.has_symbol(m->expression()->getText())) {
        power = symbol_table.get_symbol(m->expression()->getText());
        if (power.getType().isa<mlir::MemRefType>()) {
          power = builder.create<mlir::LoadOp>(location, power);
          if (power.getType().getIntOrFloatBitWidth() < 64) {
            power = builder.create<mlir::ZeroExtendIOp>(location, power,
                                                        builder.getI64Type());
          }
        }
        if (!power.getType().isIndex()) {
          power = builder.create<mlir::IndexCastOp>(
              location, builder.getIndexType(), power);
        }
      } else {
        auto p = symbol_table.evaluate_constant_integer_expression(
            m->expression()->getText());
        assert(p >= 0);
        auto pow_attr = mlir::IntegerAttr::get(builder.getIndexType(), p);
        power = builder.create<mlir::ConstantOp>(location, pow_attr);
      }
      auto powerUOp = builder.create<mlir::quantum::PowURegion>(location, power,
                                                                qbit_values);
      modifier_insertion_points_stack.push(std::make_pair(
          builder.saveInsertionPoint(), powerUOp.getOperation()));
      builder.setInsertionPointToStart(&powerUOp.body().front());
    } else if (m->getText().find("inv") != std::string::npos) {
      auto adjUOp = builder.create<mlir::quantum::AdjURegion>(location, qbit_values);
      modifier_insertion_points_stack.push(
          std::make_pair(builder.saveInsertionPoint(), adjUOp.getOperation()));
      builder.setInsertionPointToStart(&adjUOp.body().front());
    } else if (m->getText().find("ctrl") != std::string::npos) {
      mlir::Value ctrl_bit = *qbit_values.begin();
      qbit_values.erase(qbit_values.begin());
      qubit_symbol_table_keys.erase(qubit_symbol_table_keys.begin());
      auto ctrlUOp =
          builder.create<mlir::quantum::CtrlURegion>(location, ctrl_bit, qbit_values);
      modifier_insertion_points_stack.push(
          std::make_pair(builder.saveInsertionPoint(), ctrlUOp.getOperation()));
      builder.setInsertionPointToStart(&ctrlUOp.body().front());
    }
  }

  std::vector<mlir::Value> returnedValues;
  if (symbol_table.has_seen_function(name)) {
    std::vector<mlir::Value> operands;
    std::vector<mlir::Type> result_types;
    for (auto p : param_values) {
      operands.push_back(p);
    }
    for (auto q : qbit_values) {
      operands.push_back(q);
      result_types.push_back(qubit_type);
    }

    auto call_op = builder.create<mlir::CallOp>(
        location, symbol_table.get_seen_function(name), operands);

    auto return_vals = call_op.getResults();
    int i = 0;
    for (auto result : return_vals) {
      symbol_table.replace_symbol(qbit_values[i], result);
      i++;
      returnedValues.emplace_back(result);
    }

  } else {
    returnedValues = createInstOps_HandleBroadcast(name, qbit_values, qreg_names,
                                  qubit_symbol_table_keys, param_values,
                                  location, context);
  }

  // This is really not needed since the nested modifier stack 
  // only contains a single Op (could be a function call).
  // i.e., we could restore the insertion point at the bottom of the stack.
  while (!modifier_insertion_points_stack.empty()) {
    auto [insertionPt, modifierOpPtr] = modifier_insertion_points_stack.top();
    assert(returnedValues.size() == qbit_values.size());
    // For controlled modifier, we return/yield control qubit too.
    if (auto ctrlOp =
            llvm::dyn_cast_or_null<mlir::quantum::CtrlURegion>(modifierOpPtr)) {
      returnedValues.insert(returnedValues.begin(), 1, ctrlOp.ctrl_qubit());
      qbit_values.insert(qbit_values.begin(), 1, ctrlOp.ctrl_qubit());
    }
    auto modifierYieldOp = builder.create<mlir::quantum::ModifierEndOp>(location, returnedValues);
    returnedValues.clear();
    if (auto powOp = llvm::dyn_cast_or_null<mlir::quantum::PowURegion>(modifierOpPtr)) {
      assert(powOp.getResults().size() == qbit_values.size());
      for (int i = 0; i < qbit_values.size(); ++i) {
        symbol_table.replace_symbol(modifierYieldOp.getOperand(i),
                                    powOp.getResult(i));
        returnedValues.emplace_back(powOp.getResult(i));
      }
    } else if (auto adjOp =
                   llvm::dyn_cast_or_null<mlir::quantum::AdjURegion>(modifierOpPtr)) {
      assert(adjOp.getResults().size() == qbit_values.size());
      for (int i = 0; i < qbit_values.size(); ++i) {
        symbol_table.replace_symbol(modifierYieldOp.getOperand(i),
                                    adjOp.getResult(i));
        returnedValues.emplace_back(adjOp.getResult(i));
      }
    } else if (auto ctrlOp =
                   llvm::dyn_cast_or_null<mlir::quantum::CtrlURegion>(modifierOpPtr)) {
      assert(ctrlOp.getResults().size() == qbit_values.size());
      symbol_table.replace_symbol(ctrlOp.ctrl_qubit(), ctrlOp.getResult(0));
      for (int i = 0; i < qbit_values.size(); ++i) {
        symbol_table.replace_symbol(modifierYieldOp.getOperand(i),
                                    ctrlOp.getResult(i));
        returnedValues.emplace_back(ctrlOp.getResult(i));
      }
    } else {
      assert(false);
    }
    builder.restoreInsertionPoint(insertionPt);
    modifier_insertion_points_stack.pop();
  }

  return 0;
}

antlrcpp::Any qasm3_visitor::visitKernelDeclaration(
    qasm3Parser::KernelDeclarationContext* context) {
  // kernelDeclaration
  //   : 'kernel' Identifier ( LPAREN classicalTypeList? RPAREN )?
  //   returnSignature? classicalType? SEMICOLON
  //   ;
  // returnSignature
  //   : ARROW classicalType
  //   ;
  // classicalType
  //   : singleDesignatorType designator
  //   | doubleDesignatorType doubleDesignator
  //   | noDesignatorType
  //   | bitType designator?
  //   ;

  auto location = get_location(builder, file_name, context);
  auto name = context->Identifier()->getText();

  std::vector<mlir::Type> types;
  if (auto typelist = context->classicalTypeList()) {
    for (auto type_ctx : typelist->classicalType()) {
      types.push_back(convertQasm3Type(type_ctx, symbol_table, builder));
    }
  }

  bool has_return = false;
  mlir::Type return_type = builder.getIntegerType(32);
  if (context->returnSignature()) {
    has_return = true;
    auto classical_type = context->returnSignature()->classicalType();
    // can return bit, bit[], uint[], int[], float[], bool
    if (classical_type->bitType()) {
      if (auto designator = classical_type->designator()) {
        auto bit_size = symbol_table.evaluate_constant_integer_expression(
            designator->getText());
        llvm::ArrayRef<int64_t> shape{bit_size};
        return_type = mlir::MemRefType::get(shape, result_type);
      } else {
        return_type = result_type;
      }
    } else if (auto single_desig = classical_type->singleDesignatorType()) {
      if (single_desig->getText() == "float") {
        auto bit_width = symbol_table.evaluate_constant_integer_expression(
            classical_type->designator()->getText());
        if (bit_width == 16) {
          return_type = builder.getF16Type();
        } else if (bit_width == 32) {
          return_type = builder.getF32Type();
        } else if (bit_width == 64) {
          return_type = builder.getF64Type();
        } else {
          printErrorMessage(
              "on subroutine return type - we only support 16, 32, or 64 width "
              "floating point types.",
              context);
        }
      } else if (single_desig->getText() == "int") {
        auto bit_width = symbol_table.evaluate_constant_integer_expression(
            classical_type->designator()->getText());
        return_type = builder.getIntegerType(bit_width);
      } else {
        printErrorMessage(
            "we do not yet support this subroutine single designator return "
            "type.",
            context);
      }
    } else if (auto no_desig = classical_type->noDesignatorType()) {
      if (no_desig->getText().find("uint") != std::string::npos) {
        return_type = builder.getIntegerType(32, false);
      } else if (no_desig->getText().find("int64_t") != std::string::npos) {
        // This must be before "int"
        return_type = builder.getI64Type();
      } else if (no_desig->getText().find("int") != std::string::npos) {
        return_type = builder.getIntegerType(32);
      } else if (no_desig->getText().find("float") != std::string::npos) {
        return_type = builder.getF32Type();
      } else if (no_desig->getText().find("double") != std::string::npos) {
        return_type = builder.getF64Type();
      } else {
        printErrorMessage("Invalid no-designator default type.", context);
      }
    } else {
      printErrorMessage("Alex implement other return types.", context);
    }
  }
  mlir::FuncOp function;
  if (has_return) {
    auto func_type = builder.getFunctionType(types, return_type);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), name, func_type);
    mlir::FuncOp function2(proto);
    function = function2;
  } else {
    auto func_type = builder.getFunctionType(types, llvm::None);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), name, func_type);
    mlir::FuncOp function2(proto);
    function = function2;
  }

  auto savept = builder.saveInsertionPoint();

  builder.setInsertionPointToStart(&m_module.getRegion().getBlocks().front());

  // Note: MLIR FuncOp **declaration** must have non-public visibility
  // This is validated at MLIR level.
  // https://llvm.discourse.group/t/rfc-symbol-definition-declaration-x-visibility-checks/2140
  auto func_decl = builder.create<mlir::FuncOp>(
      location, name, function.getType().cast<mlir::FunctionType>());
  func_decl.setVisibility(mlir::SymbolTable::Visibility::Private);
  builder.restoreInsertionPoint(savept);

  symbol_table.add_seen_function(name, function);

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
        auto element_type =
            arg.getType().cast<mlir::MemRefType>().getElementType();
        if (!(element_type.isa<mlir::IntegerType>() &&
              element_type.getIntOrFloatBitWidth() == 1)) {
          arg = builder.create<mlir::LoadOp>(location, arg);
        }
      }
      print_args.push_back(arg);
    }
    builder.create<mlir::quantum::PrintOp>(location,
                                           llvm::makeArrayRef(print_args));
    return 0;
  }

  if (symbol_table.has_seen_function(context->Identifier()->getText())) {
    auto expr_list = context->expressionList();
    std::vector<mlir::Value> kernel_args;

    for (auto exp : expr_list->expression()) {
      // auto exp_str = exp->getText();
      qasm3_expression_generator exp_generator(builder, symbol_table,
                                               file_name);
      exp_generator.visit(exp);

      auto arg = exp_generator.current_value;
      if (arg.getType().isa<mlir::MemRefType>()) {
        arg = builder.create<mlir::LoadOp>(location, arg);
      }
      kernel_args.push_back(arg);
    }

    builder.create<mlir::CallOp>(
        location,
        symbol_table.get_seen_function(context->Identifier()->getText()),
        llvm::makeArrayRef(kernel_args));
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
      auto variable_value = qubit_exp_generator.current_value;
      if (variable_value.getType().isa<mlir::MemRefType>()) {
        if (!variable_value.getType()
                 .cast<mlir::MemRefType>()
                 .getElementType()
                 .isIntOrFloat()) {
          printErrorMessage(
              "Variable classical parameter for quantum instruction is not a "
              "float or int.",
              context, {variable_value});
        }

        auto shape =
            variable_value.getType().cast<mlir::MemRefType>().getShape();
        if (!shape.empty() && shape[0] == 1) {
          variable_value = builder.create<mlir::LoadOp>(
              location, variable_value,
              get_or_create_constant_index_value(0, location, 64, symbol_table,
                                                 builder));
        } else if (shape.empty()) {
          variable_value =
              builder.create<mlir::LoadOp>(location, variable_value);
        }
      }

      param_values.push_back(variable_value);
    }
  }

  auto str_attr = builder.getStringAttr(name);

  std::vector<std::string> qreg_names, qubit_symbol_table_keys;
  auto n_qubit_args = expression_list[qubit_expr_list_idx]->expression().size();
  for (auto expression : expression_list[qubit_expr_list_idx]->expression()) {
    auto tmp_key = expression->getText();
    tmp_key.erase(std::remove(tmp_key.begin(), tmp_key.end(), '['),
                  tmp_key.end());
    tmp_key.erase(std::remove(tmp_key.begin(), tmp_key.end(), ']'),
                  tmp_key.end());
    qreg_names.push_back(tmp_key);

    qasm3_expression_generator qubit_exp_generator(builder, symbol_table,
                                                   file_name, qubit_type);
    qubit_exp_generator.visit(expression);
    auto tmp = qubit_exp_generator.current_value;
    assert(tmp.getType().isa<mlir::OpaqueType>());
    qbit_values.push_back(tmp);
    qubit_symbol_table_keys.push_back(tmp_key);
  }

  if (symbol_table.has_seen_function(name)) {
    std::vector<mlir::Value> operands;
    for (auto p : param_values) {
      operands.push_back(p);
    }
    for (auto q : qbit_values) {
      operands.push_back(q);
    }

    builder.create<mlir::CallOp>(location, symbol_table.get_seen_function(name),
                                 operands);

  } else {
    createInstOps_HandleBroadcast(name, qbit_values, qreg_names,
                                  qubit_symbol_table_keys, param_values,
                                  location, context);
  }
  return 0;
}
}  // namespace qcor