
#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace qcor {

// classicalDeclarationStatement
//     : ( classicalDeclaration | constantDeclaration ) SEMICOLON
//     ;

// constantDeclaration
//     : 'const' equalsAssignmentList
//     ;

// singleDesignatorDeclaration
//     : singleDesignatorType designator ( identifierList | equalsAssignmentList
//     )
//     ;

// doubleDesignatorDeclaration
//     : doubleDesignatorType doubleDesignator ( identifierList |
//     equalsAssignmentList )
//     ;

// noDesignatorDeclaration
//     : noDesignatorType ( identifierList | equalsAssignmentList )
//     ;

// bitDeclaration
//     : bitType (indexIdentifierList | indexEqualsAssignmentList )
//     ;

// classicalDeclaration
//     : singleDesignatorDeclaration
//     | doubleDesignatorDeclaration
//     | noDesignatorDeclaration
//     | bitDeclaration
//     ;
//
// classicalAssignment
//     : indexIdentifier assignmentOperator ( expression | indexIdentifier )
//     ;
//
// Examples:
// const layers = 22;
// const layers2 = layers / 2;
// const t = layers * 3;
// const d = 1.2;
// const tt = d * 33.3;

// int[32] i = 10;

// const mypi = pi / 2;

// float[32] f;
// float[64] ff = 3.14;

antlrcpp::Any qasm3_visitor::visitConstantDeclaration(
    qasm3Parser::ConstantDeclarationContext* context) {
  auto ass_list = context->equalsAssignmentList();  // :)

  for (int i = 0; i < ass_list->Identifier().size(); i++) {
    std::cout << ass_list->Identifier().size() << ", "
              << ass_list->Identifier(i)->getText() << "\n";
    auto var_name = ass_list->Identifier(i)->getText();
    if (var_name == "pi") {
      printErrorMessage("pi is already defined in OPENQASM 3.");
    }
    auto equals_expr = ass_list->equalsExpression(i);

    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(equals_expr);
    auto expr_value = exp_generator.current_value;

    update_symbol_table(var_name, expr_value, {"const"});
  }

  return 0;
}

antlrcpp::Any qasm3_visitor::visitSingleDesignatorDeclaration(
    qasm3Parser::SingleDesignatorDeclarationContext* context) {
  auto location = get_location(builder, file_name, context);
  auto type = context->singleDesignatorType()->getText();
  auto designator_expr = context->designator()->expression();
  uint64_t width_idx;
  {
    // I only want the IDX in TYPE[IDX], don't need to add it the
    // the current module... This will tell me the bit width
    mlir::OpBuilder tmp_builder(builder.getContext());
    ScopedSymbolTable tmp_table;
    qasm3_expression_generator designator_exp_generator(
        tmp_builder, tmp_table, file_name, builder.getIntegerType(64));
    designator_exp_generator.visit(designator_expr);
    auto designator_value = designator_exp_generator.current_value;

    width_idx = designator_value.getDefiningOp<mlir::ConstantOp>()
                    .getValue()
                    .cast<mlir::IntegerAttr>()
                    .getUInt();
  }
  mlir::Attribute init_attr;
  mlir::Type value_type;
  if (type == "int") {
    value_type = builder.getIntegerType(width_idx);
    init_attr = mlir::IntegerAttr::get(value_type, 0);
  } else if (type == "uint") {
    value_type = builder.getIntegerType(width_idx, false);
    init_attr = mlir::IntegerAttr::get(value_type, 0);
  } else if (type == "float") {
    if (width_idx == 16) {
      value_type = mlir::FloatType::getF16(builder.getContext());
      init_attr = mlir::FloatAttr::get(value_type, 0.0);

    } else if (width_idx == 32) {
      value_type = mlir::FloatType::getF32(builder.getContext());
      init_attr = mlir::FloatAttr::get(value_type, 0.0);

    } else if (width_idx == 64) {
      value_type = mlir::FloatType::getF64(builder.getContext());
      init_attr = mlir::FloatAttr::get(value_type, 0.0);

    } else {
      printErrorMessage("we only support 16, 32, and 64 floating point types.");
    }
  } else {
    printErrorMessage("We do not currently support this type: " + type);
  }

  // THis can now be either an identifierList or an equalsAssignementList
  std::vector<std::string> variable_names;
  std::vector<mlir::Value> initial_values;
  if (auto id_list = context->identifierList()) {
    for (auto id : id_list->Identifier()) {
      variable_names.push_back(id->getText());
      initial_values.push_back(
          builder.create<mlir::ConstantOp>(location, init_attr));
    }
  } else {
    auto equals_list = context->equalsAssignmentList();
    for (auto id : equals_list->Identifier()) {
      variable_names.push_back(id->getText());
    }
    for (auto eq_expr : equals_list->equalsExpression()) {
      qasm3_expression_generator equals_exp_generator(builder, symbol_table,
                                                      file_name, value_type);
      equals_exp_generator.visit(eq_expr->expression());
      initial_values.push_back(equals_exp_generator.current_value);
    }
  }

  // Store the initial values
  for (int i = 0; i < variable_names.size(); i++) {
    auto variable = variable_names[i];
    llvm::ArrayRef<int64_t> shaperef{1};
    auto mem_type = mlir::MemRefType::get(shaperef, value_type);
    mlir::Value allocation = builder.create<mlir::AllocaOp>(location, mem_type);

    // Store the value to the 0th index of this storeop
    builder.create<mlir::StoreOp>(location, initial_values[i], allocation,
                                  get_or_create_constant_index_value(
                                      0, location, 64, symbol_table, builder));

    // Save the allocation, the store op
    symbol_table.add_symbol(variable, allocation);
  }

  return 0;
}

antlrcpp::Any qasm3_visitor::visitNoDesignatorDeclaration(
    qasm3Parser::NoDesignatorDeclarationContext* context) {
  auto location = get_location(builder, file_name, context);

  // can be no designator
  // basically can be
  // bool b;
  // bool b = 1;
  // bool b = bool(bit)
  if (context->noDesignatorType()->getText() == "bool") {
    auto eq_ass_list = context->equalsAssignmentList();
    std::vector<std::string> var_names;
    std::vector<mlir::Value> initial_values;
    mlir::Type value_type = builder.getIntegerType(1);
    if (eq_ass_list) {
      for (int i = 0; i < eq_ass_list->equalsExpression().size(); i++) {
        auto eq_expr = eq_ass_list->equalsExpression(i);
        var_names.push_back(eq_ass_list->Identifier(i)->getText());

        qasm3_expression_generator equals_exp_generator(builder, symbol_table,
                                                        file_name, value_type);
        equals_exp_generator.visit(eq_expr->expression());
        initial_values.push_back(equals_exp_generator.current_value);
      }
    } else {
      // uninitialized, just set to false;
      for (int i = 0; i < context->identifierList()->Identifier().size(); i++) {
        var_names.push_back(
            context->identifierList()->Identifier(i)->getText());
        initial_values.push_back(get_or_create_constant_integer_value(
            0, location, builder.getIntegerType(1), symbol_table, builder));
      }
    }

    for (int i = 0; i < var_names.size(); i++) {
      auto variable = var_names[i];
      llvm::ArrayRef<int64_t> shaperef{1};
      auto mem_type = mlir::MemRefType::get(shaperef, value_type);
      mlir::Value allocation =
          builder.create<mlir::AllocaOp>(location, mem_type);

      // Store the value to the 0th index of this storeop
      builder.create<mlir::StoreOp>(
          location, initial_values[i], allocation,
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));

      // Save the allocation, the store op
      symbol_table.add_symbol(variable, allocation);
    }
  }

  return 0;
}

antlrcpp::Any qasm3_visitor::visitBitDeclaration(
    qasm3Parser::BitDeclarationContext* context) {
  // bitDeclaration
  //     : bitType (indexIdentifierList | indexEqualsAssignmentList )
  //     ;
  //     indexIdentifier
  //     : Identifier rangeDefinition
  //     | Identifier ( LBRACKET expressionList RBRACKET )?
  //     | indexIdentifier '||' indexIdentifier
  //     ;

  // indexIdentifierList
  //     : ( indexIdentifier COMMA )* indexIdentifier
  //     ;
  // indexEqualsAssignmentList
  //     : ( indexIdentifier equalsExpression COMMA)* indexIdentifier
  //     equalsExpression
  //     ;
  auto location = get_location(builder, file_name, context);

  std::size_t size = 1;
  if (auto index_ident_list = context->indexIdentifierList()) {
    for (auto idx_identifier : index_ident_list->indexIdentifier()) {
      auto var_name = idx_identifier->Identifier()->getText();
      auto exp_list = idx_identifier->expressionList();
      if (exp_list) {
        size = symbol_table.evaluate_constant_integer_expression(
            exp_list->expression(0)->getText());
      }

      std::vector<mlir::Value> init_values, init_indices;
      for (std::size_t i = 0; i < size; i++) {
        init_values.push_back(get_or_create_constant_integer_value(
            0, location, builder.getI1Type(), symbol_table, builder));
        init_indices.push_back(get_or_create_constant_index_value(
            0, location, 64, symbol_table, builder));
      }
      auto allocation = allocate_1d_memory_and_initialize(
          location, size, builder.getI1Type(), init_values,
          llvm::makeArrayRef(init_indices));
      update_symbol_table(var_name, allocation);
    }
  } else {
    auto index_equals_list = context->indexEqualsAssignmentList();

    for (int i = 0; i < index_equals_list->indexIdentifier().size(); i++) {
      auto first_index_equals = index_equals_list->indexIdentifier(i);
      auto var_name = first_index_equals->Identifier()->getText();
      std::size_t size = 1;
      if (auto index_expr_list = first_index_equals->expressionList()) {
        size = symbol_table.evaluate_constant_integer_expression(
            first_index_equals->expressionList()->expression(0)->getText());
      }

      auto equals_expr =
          index_equals_list->equalsExpression()[i]->expression()->getText();
      equals_expr = equals_expr.substr(1, equals_expr.length() - 2);

      // This can only be a string-like representation of 0s and 1s
      if (size != equals_expr.length()) {
        printErrorMessage(
            "Invalid initial string assignment for bit array, sizes do not "
            "match.");
      }

      std::vector<mlir::Value> initial_values, indices;
      for (int j = 0; j < size; j++) {
        initial_values.push_back(get_or_create_constant_integer_value(
            equals_expr[j] == '1' ? 1 : 0, location, result_type, symbol_table,
            builder));
        indices.push_back(get_or_create_constant_index_value(
            j, location, 64, symbol_table, builder));
      }
      auto allocation = allocate_1d_memory_and_initialize(
          location, size, builder.getI1Type(), initial_values,
          llvm::makeArrayRef(indices));

      update_symbol_table(var_name, allocation);
    }
  }

  return 0;
}

antlrcpp::Any qasm3_visitor::visitClassicalAssignment(
    qasm3Parser::ClassicalAssignmentContext* context) {
  auto location = get_location(builder, file_name, context);
  // classicalAssignment
  //     : indexIdentifier assignmentOperator ( expression | indexIdentifier )
  //     ;

  // assignmentOperator
  //     : EQUALS
  //     | '+=' | '-=' | '*=' | '/=' | '&=' | '|=' | '~=' | '^=' | '<<=' |
  //     '>>='
  //     ;
  auto var_name = context->indexIdentifier(0)->Identifier()->getText();
  auto ass_op = context->assignmentOperator();  // :)

  if (!symbol_table.has_symbol(var_name)) {
    printErrorMessage("invalid variable name in classical assignement: " +
                      var_name + ", " + ass_op->getText());
  }

  if (!context->expression()) {
    printErrorMessage(
        "We only can handle classicalAssignment expressions at this time, "
        "no "
        "indexIdentifiers.");
  }

  if (!symbol_table.is_variable_mutable(var_name)) {
    printErrorMessage("Cannot change variable " + var_name +
                      ", it has been marked const.");
  }

  auto lhs = symbol_table.get_symbol(var_name);

  auto width = lhs.getType()
                   .cast<mlir::MemRefType>()
                   .getElementType()
                   .getIntOrFloatBitWidth();
  qasm3_expression_generator exp_generator(
      builder, symbol_table, file_name,
      lhs.getType().cast<mlir::MemRefType>().getElementType());
  exp_generator.visit(context->expression());
  auto rhs = exp_generator.current_value;

  // Could be somethign like
  // bit = subroutine_call(params) qbits...
  if (auto call_op = rhs.getDefiningOp<mlir::CallOp>()) {
    int bit_idx = 0;
    if (auto index_list = context->indexIdentifier(0)->expressionList()) {
      // Need to extract element from bit array to set it
      auto idx_str = index_list->expression(0)->getText();
      bit_idx = std::stoi(idx_str);
    }

    // Store the mz result into the bit_value
    mlir::Value pos = get_or_create_constant_integer_value(
        bit_idx, location, builder.getIntegerType(64), symbol_table, builder);

    builder.create<mlir::StoreOp>(
        location, rhs, lhs, llvm::makeArrayRef(std::vector<mlir::Value>{pos}));
    return 0;

  } else {
    if (!lhs.getType().isa<mlir::MemRefType>()) {
      printErrorMessage("cannot += to a lhs that is not a memreftype.");
    }

    llvm::ArrayRef<mlir::Value> zero_index(get_or_create_constant_index_value(
        0, location, 64, symbol_table, builder));

    auto lhs_type = lhs.getType().cast<mlir::MemRefType>().getElementType();
    mlir::Type rhs_type = rhs.getType();
    mlir::Value load_result_rhs = rhs;
    if (rhs_type.isa<mlir::MemRefType>()) {
      rhs_type = rhs_type.cast<mlir::MemRefType>().getElementType();
      auto load_rhs = builder.create<mlir::LoadOp>(location, rhs, zero_index);
      load_result_rhs = load_rhs.result();
    }

    // Load the LHS, has to load an existing allocated value
    // %2 = load i32, i32* %1
    // Create the zero int value

    auto load = builder.create<mlir::LoadOp>(location, lhs, zero_index);
    auto load_result = load.result();

    mlir::Value current_value;
    auto assignment_op = ass_op->getText();
    if (assignment_op == "+=") {
      if (lhs_type.isa<mlir::FloatType>() || rhs_type.isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::AddFOp>(location, load_result,
                                                     load_result_rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs_type.isa<mlir::IntegerType>()) {
        current_value = builder.create<mlir::AddIOp>(location, load_result,
                                                     load_result_rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);

    } else if (assignment_op == "-=") {
      if (lhs_type.isa<mlir::FloatType>() || rhs_type.isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::SubFOp>(location, load_result,
                                                     load_result_rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs_type.isa<mlir::IntegerType>()) {
        current_value = builder.create<mlir::SubIOp>(location, load_result,
                                                     load_result_rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);

    } else if (assignment_op == "*=") {
      if (lhs_type.isa<mlir::FloatType>() || rhs_type.isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::MulFOp>(location, load_result,
                                                     load_result_rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs_type.isa<mlir::IntegerType>()) {
        current_value = builder.create<mlir::MulIOp>(location, load_result,
                                                     load_result_rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);
    } else if (assignment_op == "/=") {
      if (lhs_type.isa<mlir::FloatType>() || rhs_type.isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::DivFOp>(location, load_result,
                                                     load_result_rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs_type.isa<mlir::IntegerType>()) {
        current_value = builder.create<mlir::UnsignedDivIOp>(
            location, load_result, load_result_rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);

    } else if (assignment_op == "^=") {
      current_value =
          builder.create<mlir::XOrOp>(location, load_result, load_result_rhs);
      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);
    } else {
      printErrorMessage(ass_op->getText() + " not yet supported.");
    }
  }

  return 0;
}
}  // namespace qcor