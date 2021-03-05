
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

  if (ass_list->Identifier().size() > 1) {
    printErrorMessage(
        "we only support single const variable assignement at this time.");
  }
  auto var_name = ass_list->Identifier(0)->getText();
  auto equals_expr = ass_list->equalsExpression(0);

  qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
  exp_generator.visit(equals_expr);
  auto expr_value = exp_generator.current_value;

  update_symbol_table(var_name, expr_value, {"const"});

  //   symbol_table.print();
  return 0;
}

antlrcpp::Any qasm3_visitor::visitSingleDesignatorDeclaration(
    qasm3Parser::SingleDesignatorDeclarationContext* context) {
  // Can be single designator
  // singleDesignatorType designator ( identifierList | equalsAssignmentList )
  // singleDesignatorType := int, uint, float, angle
  // designator := [expression]
  // identifierList := Identifier, Identifier, Identifier, ...
  // equalsAssignmentList := Identifier
  // examples int[10] test;
  // int[10] test = 22;

  auto location = get_location(builder, file_name, context);
  auto type = context->singleDesignatorType()->getText();
  auto designator_expr = context->designator()->expression();
  uint64_t width_idx;
  {
    // I only want the IDX in TYPE[IDX], don't need to add it the
    // the current module... This will tell me the bit width
    mlir::OpBuilder tmp_builder(builder.getContext());
    qasm3_expression_generator designator_exp_generator(
        tmp_builder, symbol_table, file_name);
    designator_exp_generator.visit(designator_expr);
    auto designator_value = designator_exp_generator.current_value;

    width_idx = designator_value.getDefiningOp<mlir::ConstantOp>()
                    .getValue()
                    .cast<mlir::IntegerAttr>()
                    .getUInt();
  }

  mlir::FloatAttr float_attr;
  mlir::Type float_type;
  if (type == "float") {
    if (width_idx == 16) {
      float_type = mlir::FloatType::getF16(builder.getContext());
      float_attr = mlir::FloatAttr::get(float_type, 0.0);

    } else if (width_idx == 32) {
      float_type = mlir::FloatType::getF32(builder.getContext());
      float_attr = mlir::FloatAttr::get(float_type, 0.0);

    } else if (width_idx == 64) {
      float_type = mlir::FloatType::getF64(builder.getContext());
      float_attr = mlir::FloatAttr::get(float_type, 0.0);

    } else {
      printErrorMessage("we only support 16, 32, and 64 floating point types.");
    }
  }

  // Create the zero int value
  mlir::Value element_zero_val =
      get_or_create_constant_integer_value(0, location, width_idx);

  mlir::Value init_val;
  std::string var_name;
  if (context->identifierList()) {
    // we have a variable, no initialization, so just set to zero

    var_name = context->identifierList()->Identifier(0)->getText();
    if (type == "int" || type == "uint") {  // FIXME hack for now
      // Initialize to 0;
      init_val =
          element_zero_val;  // builder.create<mlir::ConstantOp>(location,
                             // element_zero_attr);
    } else if (type == "float" || type == "angle") {  // FIXME hack for now
      init_val = builder.create<mlir::ConstantOp>(location, float_attr);
    }

  } else {
    // we have a variable = initial
    // set var_name and init_val
    // std::cout << "printing here " <<
    // context->equalsAssignmentList()->getText()
    //           << "\n";
    auto easslist = context->equalsAssignmentList();
    auto ids = easslist->Identifier();
    if (ids.size() > 1) {
      printErrorMessage("we only support single equal assignments.");
    }

    var_name = ids[0]->getText();
    auto equals_expr = easslist->equalsExpression(0)->expression();
    std::cout << "HI EQUALS: " << equals_expr->getText() << "\n";

    // Need to tell the expression_generator what type this
    // variable is so that it will create
    qasm3_expression_generator equals_exp_generator(
        builder, symbol_table, file_name, width_idx,
        (type == "uint" ? false : true));
    equals_exp_generator.visit(equals_expr);
    init_val = equals_exp_generator.current_value;
  }

  // create a variable allocation of given type
  mlir::Value allocation;
  if (type == "int" || type == "uint") {  // FIXME hack for now
    // Create memory allocation of dimension 0, so just a scalar
    // integer value of given width
    auto t = builder.getIntegerType(width_idx);
    if (type == "uint") {
      t = builder.getIntegerType(width_idx, false);
    }
    auto tmp = get_or_create_constant_index_value(0, location);
    allocation = allocate_1d_memory_and_initialize(
        location, 1, t, std::vector<mlir::Value>{init_val},
        llvm::makeArrayRef(std::vector<mlir::Value>{tmp}));

  } else if (type == "float" || type == "angle") {  // FIXME hack for now
    auto tmp = get_or_create_constant_index_value(0, location);
    allocation = allocate_1d_memory_and_initialize(
        location, 1, float_type, std::vector<mlir::Value>{init_val},
        llvm::makeArrayRef(std::vector<mlir::Value>{tmp}));
  } else {
    printErrorMessage("Invalid type - " + type);
  }

  // Save the allocation, the store op
  symbol_table.add_symbol(var_name, allocation);
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
    auto element_zero_attr =
        mlir::IntegerAttr::get(builder.getIntegerType(64), 0);
    auto element_false_attr = mlir::IntegerAttr::get(builder.getI1Type(), 0);

    std::string var_name;
    mlir::Value allocation, init_val;
    if (auto ident_list = context->identifierList()) {
      var_name = ident_list->Identifier(0)->getText();
      init_val = builder.create<mlir::ConstantOp>(location, element_false_attr);

      // bool b;
      auto tmp = get_or_create_constant_index_value(0, location);
      allocation = allocate_1d_memory_and_initialize(
          location, 1, builder.getI1Type(), std::vector<mlir::Value>{init_val},
          llvm::makeArrayRef(std::vector<mlir::Value>{tmp}));

    } else {
      auto equals_assignment = context->equalsAssignmentList();

      auto ids = equals_assignment->Identifier();
      if (ids.size() > 1) {
        printErrorMessage("we only support single equal assignments.");
      }
      var_name = ids[0]->getText();
      auto equals_expr = equals_assignment->equalsExpression(0)->expression();
      if (equals_expr->getText().find("bool") != std::string::npos) {
        // This is a cast expr...
        qasm3_expression_generator equals_exp_generator(builder, symbol_table,
                                                        file_name);
        equals_exp_generator.visit(equals_expr);
        init_val = equals_exp_generator.current_value;
      } else {
        std::uint64_t idx;
        {
          mlir::OpBuilder tmpbuilder(builder.getContext());
          qasm3_expression_generator equals_exp_generator(
              tmpbuilder, symbol_table, file_name);
          equals_exp_generator.visit(equals_expr);
          auto tmp_init_val = equals_exp_generator.current_value;

          if (auto tmp_op = tmp_init_val.getDefiningOp<mlir::ConstantOp>()) {
            // can set 1 or 0, so check the int value and then map to boolean i1
            idx = tmp_op.getValue().cast<mlir::IntegerAttr>().getUInt();
          }
        }

        if (idx == 1) {
          auto element_true_attr =
              mlir::IntegerAttr::get(builder.getI1Type(), 1);
          init_val =
              builder.create<mlir::ConstantOp>(location, element_true_attr);
        } else if (idx == 0) {
          init_val =
              builder.create<mlir::ConstantOp>(location, element_false_attr);
        } else {
          printErrorMessage("invalid value for bool type, must be 1 or 0.");
        }
      }

      // bool b = 1;
      auto tmp = get_or_create_constant_index_value(0, location);
      llvm::ArrayRef<mlir::Value> zero_index(tmp);
      // init_val for this cast op will be a memref, so we load
      // the value from it
      mlir::Value load_result = init_val;
      if (init_val.getType().isa<mlir::MemRefType>()) {
        auto load =
            builder.create<mlir::LoadOp>(location, init_val, zero_index);
        load_result = load.result();
      }

      // .. and then store that
      allocation = allocate_1d_memory_and_initialize(
          location, 1, builder.getI1Type(),
          std::vector<mlir::Value>{load_result},
          llvm::makeArrayRef(std::vector<mlir::Value>{tmp}));
    }

    // Add the boolean to the symbol table at current scope
    update_symbol_table(var_name, allocation);
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
        // size = std::stoi(exp_list->expression(0)->getText());

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
                  "This variable bit size must be a constant integer.");
            }
          }
        }
      }

      auto allocation = allocate_1d_memory(location, size, result_type);

      update_symbol_table(var_name, allocation);
    }
  } else {
    std::cout << context->getText() << "\n";
    auto index_equals_list = context->indexEqualsAssignmentList();
    std::cout << "HELLO WORLD: " << index_equals_list->getText() << "\n";
    if (index_equals_list->indexIdentifier().size() > 1) {
      printErrorMessage("qcor only supports single bit equal assignments.");
    }

    auto first_index_equals = index_equals_list->indexIdentifier(0);
    auto var_name = first_index_equals->Identifier()->getText();
    auto equals_expr = index_equals_list->equalsExpression()[0]->expression();

    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(equals_expr);
    auto init_val = exp_generator.current_value;

    auto tmp = get_or_create_constant_index_value(0, location);
    auto allocation = allocate_1d_memory_and_initialize(
        location, 1, builder.getI1Type(), std::vector<mlir::Value>{init_val},
        llvm::makeArrayRef(std::vector<mlir::Value>{tmp}));

    update_symbol_table(var_name, allocation);

    // printErrorMessage(
    //     "We do not yet support bit declarations with index = assignment "
    //     "list: " +
    //     context->getText());
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
  qasm3_expression_generator exp_generator(builder, symbol_table, file_name,
                                           width);
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
    mlir::Value pos = get_or_create_constant_integer_value(bit_idx, location);

    builder.create<mlir::StoreOp>(
        location, rhs, lhs, llvm::makeArrayRef(std::vector<mlir::Value>{pos}));
    return 0;

  } else {
    if (!lhs.getType().isa<mlir::MemRefType>()) {
      printErrorMessage("cannot += to a lhs that is not a memreftype.");
    }
    auto lhs_type = lhs.getType().cast<mlir::MemRefType>().getElementType();

    // Load the LHS, has to load an existing allocated value
    // %2 = load i32, i32* %1
    // Create the zero int value
    llvm::ArrayRef<mlir::Value> zero_index(
        get_or_create_constant_index_value(0, location));
    auto load = builder.create<mlir::LoadOp>(location, lhs, zero_index);
    auto load_result = load.result();

    mlir::Value current_value;
    auto assignment_op = ass_op->getText();
    if (assignment_op == "+=") {
      if (lhs_type.isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::AddFOp>(location, lhs, rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        current_value =
            builder.create<mlir::AddIOp>(location, load_result, rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);

    } else if (assignment_op == "-=") {
      if (lhs_type.isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::SubFOp>(location, lhs, rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        current_value =
            builder.create<mlir::SubIOp>(location, load_result, rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);

    } else if (assignment_op == "*=") {
      if (lhs_type.isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::MulFOp>(location, lhs, rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        current_value =
            builder.create<mlir::MulIOp>(location, load_result, rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);
    } else if (assignment_op == "/=") {
      if (lhs_type.isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        current_value = builder.create<mlir::DivFOp>(location, lhs, rhs);
      } else if (lhs_type.isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        current_value =
            builder.create<mlir::UnsignedDivIOp>(location, load_result, rhs);
      }

      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);
    } else if (assignment_op == "^=") {
      current_value = builder.create<mlir::XOrOp>(location, load_result, rhs);
      llvm::ArrayRef<mlir::Value> zero_index2(
          get_or_create_constant_index_value(0, location));
      builder.create<mlir::StoreOp>(location, current_value, lhs, zero_index2);
    } else {
      printErrorMessage(ass_op->getText() + " not yet supported.");
    }
  }

  return 0;
}
}  // namespace qcor