#include "qasm3_visitor.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitSubroutineDefinition(
    qasm3Parser::SubroutineDefinitionContext* context) {
  // : 'def' Identifier ( LPAREN classicalArgumentList? RPAREN )?
  //   quantumArgumentList? returnSignature? subroutineBlock

  // subroutineBlock
  // : LBRACE statement* returnStatement? RBRACE
  subroutine_return_statment_added = false;
  auto subroutine_name = context->Identifier()->getText();
  std::vector<mlir::Type> argument_types;
  std::vector<std::string> arg_names;
  std::vector<std::vector<std::string>> arg_attributes;
  if (auto arg_list = context->classicalArgumentList()) {
    // list of type:varname
    for (auto arg : arg_list->classicalArgument()) {
      auto type = arg->classicalType()->getText();
      argument_types.push_back(
          convertQasm3Type(arg->classicalType(), symbol_table, builder));
      arg_names.push_back(arg->association()->Identifier()->getText());
      arg_attributes.push_back({});
    }
  }

  if (context->quantumArgumentList()) {
    for (auto quantum_arg : context->quantumArgumentList()->quantumArgument()) {
      if (quantum_arg->quantumType()->getText() == "qubit" &&
          !quantum_arg->designator()) {
        arg_attributes.push_back({});
        argument_types.push_back(qubit_type);
      } else {
        argument_types.push_back(array_type);
        auto designator = quantum_arg->designator()->getText();
        if (designator == "[DYNAMIC]") {
          arg_attributes.push_back({""});
        } else {
          auto qreg_size =
              symbol_table.evaluate_constant_integer_expression(designator);
          arg_attributes.push_back({std::to_string(qreg_size)});
        }
      }

      arg_names.push_back(quantum_arg->association()->Identifier()->getText());
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
        llvm::ArrayRef<int64_t> shape(bit_size);
        return_type = mlir::MemRefType::get(shape, builder.getI1Type());
      } else {
        return_type = builder.getI1Type();
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
      } else if (no_desig->getText().find("int") != std::string::npos) {
        return_type = builder.getIntegerType(32);
      } else if (no_desig->getText().find("float") != std::string::npos) {
        return_type = builder.getF32Type();
      } else if (no_desig->getText().find("double") != std::string::npos) {
        return_type = builder.getF64Type();
      } else if (no_desig->getText().find("int64_t") != std::string::npos) {
        return_type = builder.getI64Type();
      } else {
        printErrorMessage("Invalid no-designator default type.", context);
      }
    } else {
      printErrorMessage("Alex implement other return types.", context);
    }
  }

  current_function_return_type = return_type;

  auto main_block = builder.saveInsertionPoint();

  mlir::FuncOp function, interop_function;
  if (has_return) {
    auto func_type = builder.getFunctionType(argument_types, return_type);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), subroutine_name,
                                      func_type);
    mlir::FuncOp function2(proto);
    function = function2;
  } else {
    auto func_type = builder.getFunctionType(argument_types, llvm::None);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), subroutine_name,
                                      func_type);
    mlir::FuncOp function2(proto);
    function = function2;
  }
  // Handle "extern" subroutine declaration:
  if (context->subroutineBlock()->EXTERN()) {
    // std::cout << "Handle extern subroutine: " << subroutine_name << "\n";
    builder.setInsertionPointToStart(&m_module.getRegion().getBlocks().front());
    auto func_decl = builder.create<mlir::FuncOp>(
        get_location(builder, file_name, context), subroutine_name,
        function.getType().cast<mlir::FunctionType>());
    func_decl.setVisibility(mlir::SymbolTable::Visibility::Private);
    builder.restoreInsertionPoint(main_block);
    symbol_table.add_seen_function(subroutine_name, function);
    return 0;
  }

  auto& entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  symbol_table.enter_new_scope();

  auto arguments = entryBlock.getArguments();
  for (int i = 0; i < arguments.size(); i++) {
    symbol_table.add_symbol(arg_names[i], arguments[i], arg_attributes[i]);
  }

  auto quantum_block = context->subroutineBlock();

  auto ret = visitChildren(quantum_block);

  if (!subroutine_return_statment_added) {
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
  }
  m_module.push_back(function);

  builder.restoreInsertionPoint(main_block);

  symbol_table.exit_scope();

  symbol_table.add_seen_function(subroutine_name, function);

  subroutine_return_statment_added = false;

  symbol_table.set_last_created_block(nullptr);

  // Add __interop__ function here so we can invoke from C++.
  mlir::FunctionType interop_func_type;
  // QASM3 subroutines have classical args followed by qubit args
  // for qcor interop, we need qubit/qreg arg first.
  // FIXME Only do this for subroutines with a single qubit[] array.
  std::reverse(argument_types.begin(), argument_types.end());
  if (has_return) {
    interop_func_type = builder.getFunctionType(argument_types, return_type);
  } else {
    interop_func_type = builder.getFunctionType(argument_types, llvm::None);
  }

  auto interop =
      mlir::FuncOp::create(builder.getUnknownLoc(),
                           subroutine_name + "__interop__", interop_func_type);
  auto& interop_entryBlock = *interop.addEntryBlock();
  builder.setInsertionPointToStart(&interop_entryBlock);

  std::vector<mlir::BlockArgument> vec_to_reverse;
  for (int i = 0; i < arguments.size(); i++) {
    vec_to_reverse.push_back(interop_entryBlock.getArgument(i));
  }
  std::reverse(vec_to_reverse.begin(), vec_to_reverse.end());

  auto call_op_interop = builder.create<mlir::CallOp>(
      builder.getUnknownLoc(), function, llvm::makeArrayRef(vec_to_reverse));
  if (has_return) {
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                   call_op_interop.getResults());
  } else {
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
  }

  builder.restoreInsertionPoint(main_block);

  m_module.push_back(interop);
  return 0;
}

antlrcpp::Any qasm3_visitor::visitReturnStatement(
    qasm3Parser::ReturnStatementContext* context) {
  is_return_stmt = true;
  auto location = get_location(builder, file_name, context);

  auto ret_stmt = context->statement()->getText();
  ret_stmt.erase(std::remove(ret_stmt.begin(), ret_stmt.end(), ';'),
                 ret_stmt.end());

  mlir::Value value;
  if (symbol_table.has_symbol(ret_stmt)) {
    value = symbol_table.get_symbol(ret_stmt);
    // Actually return value if it is a bit[],
    // load and return if it is a bit

    if (current_function_return_type) {  // this means it is a subroutine
      if (!current_function_return_type.isa<mlir::MemRefType>()) {
        if (current_function_return_type.isa<mlir::IntegerType>() &&
            current_function_return_type.getIntOrFloatBitWidth() == 1) {
          // This is a bit and not a bit[]
          auto tmp = get_or_create_constant_index_value(0, location, 64,
                                                        symbol_table, builder);
          llvm::ArrayRef<mlir::Value> zero_index(tmp);
          if (value.getType().isa<mlir::MemRefType>() &&
              value.getType().cast<mlir::MemRefType>().getShape().empty()) {
            // No need to add index if the Memref has no dimension.
            value = builder.create<mlir::LoadOp>(location, value);
          } else {
            value = builder.create<mlir::LoadOp>(location, value, zero_index);
          }
        } else {
          value =
              builder.create<mlir::LoadOp>(location, value);  //, zero_index);
        }
      } else {
        printErrorMessage("We do not return memrefs from subroutines.",
                          context);
      }
    } else {
      if (auto t = value.getType().dyn_cast_or_null<mlir::MemRefType>()) {
        if (t.getRank() == 0) {
          value = builder.create<mlir::LoadOp>(location, value);
        }
      }
    }

  } else {
    if (auto expr_stmt = context->statement()->expressionStatement()) {
      qasm3_expression_generator exp_generator(builder, symbol_table,
                                               file_name);
      exp_generator.visit(expr_stmt);
      value = exp_generator.current_value;
    } else {
      visitChildren(context->statement());
      value = symbol_table.get_last_value_added();
    }
  }
  is_return_stmt = false;

  builder.create<mlir::ReturnOp>(
      builder.getUnknownLoc(),
      llvm::makeArrayRef(std::vector<mlir::Value>{value}));
  subroutine_return_statment_added = true;
  return 0;
}
}  // namespace qcor