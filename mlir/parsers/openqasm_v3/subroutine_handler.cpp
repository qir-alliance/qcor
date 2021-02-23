#include "qasm3_visitor.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitSubroutineDefinition(
    qasm3Parser::SubroutineDefinitionContext* context) {
  // : 'def' Identifier ( LPAREN classicalArgumentList? RPAREN )?
  //   quantumArgumentList? returnSignature? subroutineBlock

  // subroutineBlock
  // : LBRACE statement* returnStatement? RBRACE
  auto subroutine_name = context->Identifier()->getText();
  std::vector<mlir::Type> argument_types;
  std::vector<std::string> arg_names;
  if (auto arg_list = context->classicalArgumentList()) {
    // list of type:varname
    for (auto arg : arg_list->classicalArgument()) {
      //       bitType
      //     : 'bit'
      //     | 'creg'
      //     ;

      // singleDesignatorType
      //     : 'int'
      //     | 'uint'
      //     | 'float'
      //     | 'angle'
      //     ;

      // doubleDesignatorType
      //     : 'fixed'
      //     ;

      // noDesignatorType
      //     : 'bool'
      //     | timingType
      //     ;

      // classicalType
      //     : singleDesignatorType designator
      //     | doubleDesignatorType doubleDesignator
      //     | noDesignatorType
      //     | bitType designator?
      //     ;
      auto type = arg->classicalType()->getText();
      if (type == "bit") {
        // result type
        argument_types.push_back(result_type);
      } else if (type.find("bit") != std::string::npos &&
                 type.find("[") != std::string::npos) {
        // array type
        auto start = type.find_first_of("[");
        auto finish = type.find_first_of("]");
        auto idx_str = type.substr(start + 1, finish - start - 1);

        auto is_constant = [](const std::string idx_str) {
          try {
            std::stoi(idx_str);
            return true;
          } catch (std::exception& ex) {
            return false;
          }
        };

        mlir::Type mlir_type;
        if (is_constant(idx_str)) {
          auto bit_size = std::stoi(idx_str);
          llvm::ArrayRef<int64_t> shaperef{bit_size};
          mlir_type = mlir::MemRefType::get(shaperef, result_type);
        } else {
          // this is a variable idx size
          if (!symbol_table.has_symbol(idx_str)) {
            printErrorMessage(idx_str +
                              " is not an allocated variable or constant.");
          }
          auto value = symbol_table.get_symbol(idx_str);
          auto op = value.getDefiningOp<mlir::ConstantOp>();
          if (op && op.getValue().isa<mlir::IntegerAttr>()) {
            auto bit_size = op.getValue().cast<mlir::IntegerAttr>().getInt();
            llvm::ArrayRef<int64_t> shaperef{bit_size};
            mlir_type = mlir::MemRefType::get(shaperef, result_type);
          } else {
            printErrorMessage(
                "The bit index for this subroutine argument must be a constant "
                "integer type.");
          }
        }

        argument_types.push_back(mlir_type);

      } else if (type == "bool") {
      } else if (type.find("uint") != std::string::npos) {
      } else if (type.find("int") != std::string::npos) {
      }

      arg_names.push_back(arg->association()->Identifier()->getText());
    }
  }

  if (context->quantumArgumentList()) {
    for (auto quantum_arg : context->quantumArgumentList()->quantumArgument()) {
      if (quantum_arg->quantumType()->getText() == "qubit") {
        argument_types.push_back(qubit_type);
      } else {
        argument_types.push_back(array_type);
      }
      arg_names.push_back(quantum_arg->association()->Identifier()->getText());
    }
  }

  mlir::Type return_type;
  if (context->returnSignature()) {
    auto classical_type = context->returnSignature()->classicalType();
    // can return bit, bit[], uint[], int[], float[], bool
    if (classical_type->bitType()) {
      return_type = result_type;
    }
  }

  current_function_return_type = return_type;
  
  auto main_block = builder.saveInsertionPoint();

  auto func_type = builder.getFunctionType(argument_types, return_type);
  auto proto =
      mlir::FuncOp::create(builder.getUnknownLoc(), subroutine_name, func_type);
  mlir::FuncOp function(proto);
  auto& entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  symbol_table.enter_new_scope();

  auto arguments = entryBlock.getArguments();
  for (int i = 0; i < arguments.size(); i++) {
    symbol_table.add_symbol(arg_names[i], arguments[i]);
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
    // printErrorMessage("Putting this here til I fix this");
    if (!current_function_return_type.isa<mlir::MemRefType>()) {
      // This is a bit and not a bit[]
      auto tmp = get_or_create_constant_index_value(0, location);
      llvm::ArrayRef<mlir::Value> zero_index(tmp);
      value = builder.create<mlir::LoadOp>(location, value, zero_index);
    }
    
  } else {
    visitChildren(context->statement());

    value = symbol_table.get_last_value_added();
  }
  is_return_stmt = false;

  builder.create<mlir::ReturnOp>(
      builder.getUnknownLoc(),
      llvm::makeArrayRef(std::vector<mlir::Value>{value}));
  subroutine_return_statment_added = true;
  return 0;
}
}  // namespace qcor