#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitAliasStatement(
    qasm3Parser::AliasStatementContext* context) {
  auto location = get_location(builder, file_name, context);

  /** Aliasing **/
  // aliasStatement
  //     : 'let' Identifier EQUALS indexIdentifier SEMICOLON
  //     ;

  // /** Register Concatenation and Slicing **/

  // indexIdentifier
  //     : Identifier rangeDefinition
  //     | Identifier ( LBRACKET expressionList RBRACKET )?
  //     | indexIdentifier '||' indexIdentifier
  //     ;

  printErrorMessage(
      "Alias not yet supported yet - Thien, remove this for development.");

  // The name of the new alias register, pointing to previously allocated
  // register
  auto alias = context->Identifier()->getText();

  // Get the name and symbol Value of the original register
  auto allocated_variable = context->indexIdentifier()->Identifier()->getText();
  auto allocated_symbol = symbol_table.get_symbol(allocated_variable);

  // handle q[1, 3, 5] comma syntax
  if (context->indexIdentifier()->LBRACKET()) {
    // get the comma expression, count how many elements there are
    auto expressions =
        context->indexIdentifier()->expressionList()->expression();
    auto n_expressions = expressions.size();

    // Allocate a new array of qubits of given size
    auto str_attr = builder.getStringAttr(alias);
    auto integer_attr =
        mlir::IntegerAttr::get(builder.getI64Type(), n_expressions);
    mlir::Value allocation = builder.create<mlir::quantum::QallocOp>(
        location, array_type, integer_attr, str_attr);

    auto counter = 0;
    for (auto expr : expressions) {
      // GOAL HERE IS TO ASSIGN extracted qubits from original array
      // to the correct element of the alias array
      auto idx =
          symbol_table.evaluate_constant_integer_expression(expr->getText());

      // get the extracted element from the original register
      auto extracted = builder.create<mlir::quantum::ExtractQubitOp>(
          location, qubit_type, allocated_symbol,
          get_or_create_constant_integer_value(
              idx, location, builder.getI64Type(), symbol_table, builder));

      // use extracted with a new qassign dialect operation.
    }

  } else if (auto range_def = context->indexIdentifier()->rangeDefinition()) {
    // handle range definition
    // I think we can handle RANGE with a memref<3xi64>...

  } else {
    // handle concatenation
  }
  return 0;
}
}  // namespace qcor