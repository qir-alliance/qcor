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

      } else if (type.find("bit") != std::string::npos &&
                 type.find("[") != std::string::npos) {
        // array type
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
  visitChildren(context->statement());
  is_return_stmt = false;

  auto value = symbol_table.get_last_value_added();
  builder.create<mlir::ReturnOp>(
      builder.getUnknownLoc(),
      llvm::makeArrayRef(std::vector<mlir::Value>{value}));
  subroutine_return_statment_added = true;
}
}  // namespace qcor