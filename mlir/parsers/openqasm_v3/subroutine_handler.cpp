#include "qasm3_visitor.hpp"

namespace qcor {
    antlrcpp::Any qasm3_visitor::visitSubroutineDefinition(
      qasm3Parser::SubroutineDefinitionContext* context)  {
    // : 'def' Identifier ( LPAREN classicalArgumentList? RPAREN )?
    //   quantumArgumentList? returnSignature? subroutineBlock

    // subroutineBlock
    // : LBRACE statement* returnStatement? RBRACE
    auto subroutine_name = context->Identifier()->getText();
    std::vector<mlir::Type> argument_types;
    std::vector<std::string> arg_names;
    if (context->classicalArgumentList()) {
      // list of type:varname
    }

    if (context->quantumArgumentList()) {

      for (auto quantum_arg : context->quantumArgumentList()->quantumArgument()) {
        if (quantum_arg->quantumType()->getText() == "qubit") {
          argument_types.push_back(qubit_type);
        } else {
          argument_types.push_back(array_type);
        }
      }

    }

    mlir::Type return_type;
    if (context->returnSignature()) {
      auto classical_type = context->returnSignature()->classicalType();
      // can return bit, bit[], uint[], int[], float[], bool
      if (classical_type->bitType()) {

      }
    }

    auto main_block = builder.saveInsertionPoint();

    auto func_type = builder.getFunctionType(argument_types, return_type);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), subroutine_name,
                                      func_type);
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

    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    m_module.push_back(function);

    builder.restoreInsertionPoint(main_block);

    // current_local_scope_symbol_table.clear();
    symbol_table.exit_scope();

    seen_functions.insert({subroutine_name, function});

    return 0;
  }
}