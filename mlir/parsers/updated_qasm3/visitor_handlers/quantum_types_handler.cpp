#include "qasm3_visitor.hpp"

namespace qcor {

antlrcpp::Any qasm3_visitor::visitQuantumDeclaration(
    qasm3Parser::QuantumDeclarationContext* context) {
  // quantumDeclaration
  //     : quantumType indexIdentifierList
  //     ;
  //
  //   indexIdentifier
  //     : Identifier rangeDefinition
  //     | Identifier ( LBRACKET expressionList RBRACKET )?
  //     | indexIdentifier '||' indexIdentifier
  //     ;

  // indexIdentifierList
  //     : ( indexIdentifier COMMA )* indexIdentifier
  //     ;
  //
  // can be
  // qubit q;
  // qubit q[size];
  // qubit q[size], r[size2], ...;
  // qreg q[size];

  auto location = get_location(builder, file_name, context);

  std::size_t size = 1;
  auto index_ident_list = context->indexIdentifierList();

  for (auto idx_identifier : index_ident_list->indexIdentifier()) {
    auto var_name = idx_identifier->Identifier()->getText();
    auto exp_list = idx_identifier->expressionList();
    if (exp_list) {
      try {
        size = std::stoi(exp_list->expression(0)->getText());
      } catch (...) {
        // check if this is a constant expression
        qasm3_expression_generator exp_generator(builder, symbol_table,
                                                 file_name);
        exp_generator.visit(exp_list->expression(0));
        auto arg = exp_generator.current_value;

        if (auto constantOp = arg.getDefiningOp<mlir::ConstantOp>()) {
          if (constantOp.getValue().isa<mlir::IntegerAttr>()) {
            size = constantOp.getValue().cast<mlir::IntegerAttr>().getInt();
          } else {
            printErrorMessage(
                "This variable qubit size must be a constant integer.");
          }
        }
      }
    }

    auto integer_type = builder.getI64Type();
    auto integer_attr = mlir::IntegerAttr::get(integer_type, size);

    auto str_attr = builder.getStringAttr(var_name);
    mlir::Value allocation = builder.create<mlir::quantum::QallocOp>(
        location, array_type, integer_attr, str_attr);

    if (context->quantumType()->getText() == "qubit" && size == 1) {
      // we have a single qubit, dont set it as an array in teh
      // symbol table, extract it and set it
      mlir::Value pos = get_or_create_constant_integer_value(
          0, location, builder.getIntegerType(64), symbol_table, builder);

      // Need to also store the qubit array for this single qubit
      // so that we can deallocate later.
      update_symbol_table("__qcor__mlir__single_qubit_register_" + var_name,
                          allocation);

      allocation = builder.create<mlir::quantum::ExtractQubitOp>(
          location, qubit_type, allocation, pos);
    }

    update_symbol_table(var_name, allocation);
    size = 1;
  }
  return 0;
}

}  // namespace qcor