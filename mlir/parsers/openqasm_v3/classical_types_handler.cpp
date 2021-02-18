
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
  uint64_t idx;
  {
    // I only want the IDX in TYPE[IDX], don't need to add it the
    // the current module... This will tell me the bit width
    mlir::OpBuilder tmp_builder(builder.getContext());
    qasm3_expression_generator designator_exp_generator(
        tmp_builder, symbol_table, file_name);
    designator_exp_generator.visit(designator_expr);
    auto designator_value = designator_exp_generator.current_value;

    idx = designator_value.getDefiningOp<mlir::ConstantOp>()
              .getValue()
              .cast<mlir::IntegerAttr>()
              .getUInt();
  }
//   std::cout << "HELLO: " << idx << "\n";

  auto element_zero_attr =
      mlir::IntegerAttr::get(builder.getIntegerType(idx), 0);
  mlir::FloatAttr float_attr;
  mlir::Type float_type;
  if (idx == 16) {
    float_type = mlir::FloatType::getF16(builder.getContext());
    float_attr = mlir::FloatAttr::get(float_type, 0.0);

  } else if (idx == 32) {
    float_type = mlir::FloatType::getF32(builder.getContext());
    float_attr = mlir::FloatAttr::get(float_type, 0.0);

  } else if (idx == 64) {
    float_type = mlir::FloatType::getF64(builder.getContext());
    float_attr = mlir::FloatAttr::get(float_type, 0.0);

  } else {
    printErrorMessage("we only support 16, 32, and 64 floating point types.");
  }
  mlir::Value element_zero_val;
  if (symbol_table.has_constant_integer(0)) {
    element_zero_val = symbol_table.get_constant_integer(0);
  } else {
    element_zero_val = create_constant_integer_value(0, location);
  }
  //   builder.create<mlir::ConstantOp>(location, element_zero_attr);

  mlir::Value init_val;
  std::string var_name;
  if (context->identifierList()) {
    // we have a variable, no initialization
    var_name = context->identifierList()->Identifier(0)->getText();
    // std::cout << "varname; " << var_name << "\n";

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
    // std::cout << "printing here " << context->equalsAssignmentList()->getText()
    //           << "\n";
    auto easslist = context->equalsAssignmentList();
    auto ids = easslist->Identifier();
    if (ids.size() > 1) {
      printErrorMessage("we only support single equal assignments.");
    }
    var_name = ids[0]->getText();
    auto equals_expr = easslist->equalsExpression(0)->expression();
    qasm3_expression_generator equals_exp_generator(builder, symbol_table,
                                                    file_name);
    equals_exp_generator.visit(equals_expr);
    init_val = equals_exp_generator.current_value;
  }

  // create a variable allocation of given type
  llvm::ArrayRef<int64_t> shape{1};
  mlir::Value allocation;
  if (type == "int" || type == "uint") {  // FIXME hack for now
    auto int_type = mlir::MemRefType::get(shape, builder.getIntegerType(idx));
    allocation = builder.create<mlir::AllocOp>(location, int_type);
    builder.create<mlir::StoreOp>(
        location, init_val, allocation,
        llvm::makeArrayRef(std::vector<mlir::Value>{element_zero_val}));
  } else if (type == "float" || type == "angle") {  // FIXME hack for now
    auto float_type_mem = mlir::MemRefType::get(shape, float_type);
    allocation = builder.create<mlir::AllocOp>(location, float_type_mem);
    builder.create<mlir::StoreOp>(
        location, init_val, allocation,
        llvm::makeArrayRef(std::vector<mlir::Value>{element_zero_val}));
  } else {
    printErrorMessage("Invalid type - " + type);
  }

  symbol_table.add_symbol(var_name, allocation);

//   std::cout << "IS ALLOCATION: " << var_name << ", "
//             << symbol_table.is_allocation(var_name) << "\n";
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

    mlir::Value element_zero_val;
    if (symbol_table.has_constant_integer(0)) {
      element_zero_val = symbol_table.get_constant_integer(0);
    } else {
      element_zero_val = create_constant_integer_value(0, location);
    }
    // auto element_zero_val =
    // builder.create<mlir::ConstantOp>(location, element_zero_attr);

    std::string var_name;
    llvm::ArrayRef<int64_t> shape{1};
    auto bool_type = mlir::MemRefType::get(shape, builder.getI1Type());
    mlir::Value allocation, init_val;
    if (auto ident_list = context->identifierList()) {
      var_name = ident_list->Identifier(0)->getText();
      init_val = builder.create<mlir::ConstantOp>(location, element_false_attr);
      // bool b;
      allocation = builder.create<mlir::AllocOp>(location, bool_type);
      builder.create<mlir::StoreOp>(
          location, init_val, allocation,
          llvm::makeArrayRef(std::vector<mlir::Value>{element_zero_val}));

    } else {
      auto equals_assignment = context->equalsAssignmentList();

      auto ids = equals_assignment->Identifier();
      if (ids.size() > 1) {
        printErrorMessage("we only support single equal assignments.");
      }
      var_name = ids[0]->getText();
      auto equals_expr = equals_assignment->equalsExpression(0)->expression();
      std::uint64_t idx;
      {
        mlir::OpBuilder tmpbuilder(builder.getContext());
        qasm3_expression_generator equals_exp_generator(
            tmpbuilder, symbol_table, file_name);
        equals_exp_generator.visit(equals_expr);
        auto tmp_init_val = equals_exp_generator.current_value;

        // can set 1 or 0, so check the int value and then map to boolean i1
        idx = tmp_init_val.getDefiningOp<mlir::ConstantOp>()
                  .getValue()
                  .cast<mlir::IntegerAttr>()
                  .getUInt();
      }
      if (idx == 1) {
        auto element_true_attr = mlir::IntegerAttr::get(builder.getI1Type(), 1);
        init_val =
            builder.create<mlir::ConstantOp>(location, element_true_attr);
      } else if (idx == 0) {
        init_val =
            builder.create<mlir::ConstantOp>(location, element_false_attr);
      } else {
        printErrorMessage("invalid value for bool type, must be 1 or 0.");
      }

      // bool b = 1;
      allocation = builder.create<mlir::AllocOp>(location, bool_type);
      builder.create<mlir::StoreOp>(
          location, init_val, allocation,
          llvm::makeArrayRef(std::vector<mlir::Value>{element_zero_val}));
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
  auto location = get_location(builder, file_name, context);

  std::size_t size = 1;
  if (auto index_ident_list = context->indexIdentifierList()) {
    for (auto idx_identifier : index_ident_list->indexIdentifier()) {
      auto var_name = idx_identifier->Identifier()->getText();
      auto exp_list = idx_identifier->expressionList();
      if (exp_list) {
        size = std::stoi(exp_list->expression(0)->getText());
      }

      auto integer_type = builder.getI64Type();
      auto integer_attr = mlir::IntegerAttr::get(integer_type, size);

      auto str_attr = builder.getStringAttr(var_name);
      mlir::Value allocation = builder.create<mlir::quantum::ResultAllocOp>(
          location, array_type, integer_attr, str_attr);

      if (context->bitType()->getText() == "bit" && size == 1) {
        // we have a single cbit, dont set it as an array in the
        // symbol table, extract it and set it
        mlir::Value pos;
        if (symbol_table.has_constant_integer(0)) {
          pos = symbol_table.get_constant_integer(0);
        } else {
          pos = create_constant_integer_value(0, location);
        }

        // auto pos = get_constant_integer_idx(0, location);
        allocation = builder.create<mlir::quantum::ExtractCbitOp>(
            location, qubit_type, allocation, pos);
      }

      update_symbol_table(var_name, allocation);
    }
  } else {
    auto index_equals_list = context->indexEqualsAssignmentList();
    printErrorMessage(
        "We do not yet support bit declarations with index = assignment list");
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

  std::cout << "TESTING " << var_name << ", " << ass_op->getText() << "\n";

  qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
  exp_generator.visit(context->expression());
  auto rhs = exp_generator.current_value;
  auto lhs = symbol_table.get_symbol(var_name);

  auto lhs_ptr = builder.create<mlir::LoadOp>(location, lhs);
  mlir::Value current_value;
  auto assignment_op = ass_op->getText();
  if (assignment_op == "+=") {
    if (lhs.getType().isa<mlir::FloatType>() ||
        rhs.getType().isa<mlir::FloatType>()) {
      current_value = builder.create<mlir::AddFOp>(location, lhs, rhs);
    } else if (lhs.getType().isa<mlir::IntegerType>() &&
               rhs.getType().isa<mlir::IntegerType>()) {
      current_value = builder.create<mlir::AddIOp>(location, lhs, rhs);
    }
    builder.create<mlir::StoreOp>(location, current_value, lhs_ptr);
  }

  // if (ass_list->Identifier().size() > 1) {
  //   printErrorMessage(
  //       "we only support single const variable assignement at this
  //       time.");
  // }
  // auto var_name = ass_list->Identifier(0)->getText();
  // auto equals_expr = ass_list->equalsExpression(0);

  // qasm3_expression_generator exp_generator(builder, symbol_table,
  // file_name); exp_generator.visit(equals_expr); auto expr_value =
  // exp_generator.current_value;

  // update_symbol_table(var_name, expr_value);
  return 0;
}
}  // namespace qcor