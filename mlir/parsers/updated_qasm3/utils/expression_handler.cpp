
#include "expression_handler.hpp"
using namespace qasm3;

namespace qcor {

void qasm3_expression_generator::update_current_value(mlir::Value v) {
  last_current_value = current_value;
  current_value = v;
  return;
}
qasm3_expression_generator::qasm3_expression_generator(mlir::OpBuilder b,
                                                       ScopedSymbolTable& table,
                                                       std::string& fname,
                                                       mlir::Type t)
    : builder(b),
      file_name(fname),
      symbol_table(table),
      internal_value_type(t) {}

qasm3_expression_generator::qasm3_expression_generator(mlir::OpBuilder b,
                                                       ScopedSymbolTable& table,
                                                       std::string& fname,
                                                       std::size_t nw,
                                                       bool is_s)
    : builder(b),
      file_name(fname),
      symbol_table(table),
      number_width(nw),
      is_signed(is_s) {}

antlrcpp::Any qasm3_expression_generator::visitTerminal(
    antlr4::tree::TerminalNode* node) {
  auto location = builder.getUnknownLoc();  //(builder, file_name, ctx);
  if (node->getSymbol()->getText() == "[") {
    // We have hit a closing on an index
    // std::cout << "TERMNODE:\n";
    indexed_variable_value = current_value;
    if (casting_indexed_integer_to_bool) {
      internal_value_type = builder.getIndexType();
    }
  } else if (node->getSymbol()->getText() == "]") {
    if (casting_indexed_integer_to_bool) {
      // We have an indexed integer in indexed_variable_value
      // We want to get its idx bit and set that as the
      // current value so that we can cast it to a bool
      // need to code up the following
      // ((NUMBER >> (IDX-1)) & 1)
      // shift_right then AND 1

      // CASE
      // uint[4] b_in = 15; // b = 1111
      // bool(b_in[1]);

      // auto number_value = builder.create<mlir::LoadOp>(location,
      // indexed_variable_value, get_or_create_constant_index_value(0,
      // location)); number_value.dump(); auto idx_minus_1 =
      // builder.create<mlir::SubIOp>(location, current_value,
      // get_or_create_constant_integer_value(1, location));
      auto bw = indexed_variable_value.getType().getIntOrFloatBitWidth();
      auto casted_idx =
          builder.create<mlir::IndexCastOp>(location, current_value,
                                            indexed_variable_value.getType()
                                                .cast<mlir::MemRefType>()
                                                .getElementType());
      auto load_value = builder.create<mlir::LoadOp>(
          location, indexed_variable_value,
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
      auto shift = builder.create<mlir::UnsignedShiftRightOp>(
          location, load_value, casted_idx);
      // auto shift_load_value = builder.create<mlir::LoadOp>(
      //     location, shift,
      //     get_or_create_constant_index_value(0, location, 64, symbol_table,
      //                                        builder));
      auto old_int_type = internal_value_type;
      internal_value_type = indexed_variable_value.getType();
      auto and_value = builder.create<mlir::AndOp>(
          location, shift,
          get_or_create_constant_integer_value(1, location,
                                               indexed_variable_value.getType()
                                                   .cast<mlir::MemRefType>()
                                                   .getElementType(),
                                               symbol_table, builder));
      internal_value_type = old_int_type;
      update_current_value(and_value.result());
      casting_indexed_integer_to_bool = false;
    } else {
      if (internal_value_type.dyn_cast_or_null<mlir::OpaqueType>() &&
          internal_value_type.cast<mlir::OpaqueType>().getTypeData().str() ==
              "Qubit") {
        if (current_value.getType().isa<mlir::MemRefType>()) {
          if (current_value.getType().cast<mlir::MemRefType>().getRank() == 1 &&
              current_value.getType().cast<mlir::MemRefType>().getShape()[0] ==
                  1) {
            current_value = builder.create<mlir::LoadOp>(
                location, current_value,
                get_or_create_constant_index_value(0, location, 64,
                                                   symbol_table, builder));
          } else {
            printErrorMessage("Terminator ']' -> Invalid qubit array index: ",
                              current_value);
          }
        }
        update_current_value(builder.create<mlir::quantum::ExtractQubitOp>(
            location, get_custom_opaque_type("Qubit", builder.getContext()),
            indexed_variable_value, current_value));
      } else {
        // We are loading from a variable
        llvm::ArrayRef<mlir::Value> idx(current_value);
        update_current_value(builder.create<mlir::LoadOp>(
            location, indexed_variable_value, idx));
      }
    }
  }
  return 0;
}

antlrcpp::Any qasm3_expression_generator::visitExpression(
    qasm3Parser::ExpressionContext* ctx) {
  return visitChildren(ctx);
}
antlrcpp::Any qasm3_expression_generator::visitComparsionExpression(
    qasm3Parser::ComparsionExpressionContext* compare) {
  auto location = get_location(builder, file_name, compare);

  if (auto relational_op = compare->relationalOperator()) {
    visitChildren(compare->expression(0));
    auto lhs = current_value;
    visitChildren(compare->expression(1));
    auto rhs = current_value;

    // if lhs is memref of rank 1 and size 1, this is a
    // variable and we need to load its value
    auto lhs_type = lhs.getType();
    auto rhs_type = rhs.getType();
    if (auto mem_value_type = lhs_type.dyn_cast_or_null<mlir::MemRefType>()) {
      if (mem_value_type.getElementType().isIntOrIndex() &&
          mem_value_type.getRank() == 1 && mem_value_type.getShape()[0] == 1) {
        // Load this memref value

        lhs = builder.create<mlir::LoadOp>(
            location, lhs,
            get_or_create_constant_index_value(0, location, 64, symbol_table,
                                               builder));
      }
    }

    if (auto mem_value_type = rhs_type.dyn_cast_or_null<mlir::MemRefType>()) {
      if (mem_value_type.getElementType().isIntOrIndex() &&
          mem_value_type.getRank() == 1 && mem_value_type.getShape()[0] == 1) {
        // Load this memref value

        rhs = builder.create<mlir::LoadOp>(
            location, rhs,
            get_or_create_constant_index_value(0, location, 64, symbol_table,
                                               builder));
      }
    }

    auto op = relational_op->getText();
    if (antlr_to_mlir_predicate.count(op)) {
      // if so, get the mlir enum representing it
      auto predicate = antlr_to_mlir_predicate[op];

      auto lhs_bw = lhs.getType().getIntOrFloatBitWidth();
      auto rhs_bw = rhs.getType().getIntOrFloatBitWidth();
      // We need the comparison to be on the same bit width
      if (lhs_bw < rhs_bw) {
        rhs = builder.create<mlir::IndexCastOp>(location, rhs,
                                                builder.getIntegerType(lhs_bw));
      } else if (lhs_bw > rhs_bw) {
        lhs = builder.create<mlir::IndexCastOp>(location, lhs,
                                                builder.getIntegerType(rhs_bw));
      }

      // create the binary op value
      update_current_value(
          builder.create<mlir::CmpIOp>(location, predicate, lhs, rhs));
      return 0;
    } else {
      printErrorMessage("Invalid relational operation: " + op);
    }

  } else {
    // This is just if(expr)
    // printErrorMessage("Alex please implement if(expr).");

    found_negation_unary_op = false;
    visitChildren(compare->expression(0));
    // now just compare current_value to 1
    mlir::Type current_value_type =
        current_value.getType().isa<mlir::MemRefType>()
            ? current_value.getType().cast<mlir::MemRefType>().getElementType()
            : current_value.getType();

    current_value = builder.create<mlir::LoadOp>(
        location, current_value,
        get_or_create_constant_index_value(0, location, 64, symbol_table,
                                           builder));

    mlir::CmpIPredicate p = mlir::CmpIPredicate::eq;
    if (found_negation_unary_op) {
      p = mlir::CmpIPredicate::ne;
    }

    current_value = builder.create<mlir::CmpIOp>(
        location, p, current_value,
        get_or_create_constant_integer_value(1, location, current_value_type,
                                             symbol_table, builder));
    return 0;
  }
  return visitChildren(compare);
}

antlrcpp::Any qasm3_expression_generator::visitBooleanExpression(
    qasm3Parser::BooleanExpressionContext* ctx) {
  auto location = get_location(builder, file_name, ctx);

  if (ctx->logicalOperator()) {
    auto bool_expr = ctx->booleanExpression();
    visitChildren(bool_expr);
    auto lhs = current_value;

    visit(ctx->comparsionExpression());
    auto rhs = current_value;

    if (ctx->logicalOperator()->getText() == "&&") {
      update_current_value(builder.create<mlir::AndOp>(location, lhs, rhs));
      return 0;
    }
  }
  return visitChildren(ctx);
}

antlrcpp::Any qasm3_expression_generator::visitUnaryExpression(
    qasm3Parser::UnaryExpressionContext* ctx) {
  if (auto unary_op = ctx->unaryOperator()) {
    if (unary_op->getText() == "!") {
      found_negation_unary_op = true;
    }
  }
  return visitChildren(ctx);
}
// antlrcpp::Any qasm3_expression_generator::visitIncrementor(
//     qasm3Parser::IncrementorContext* ctx) {
//   auto location = get_location(builder, file_name, ctx);

//   auto type = ctx->getText();
//   if (type == "++") {
//     if (current_value.getType().isa<mlir::IntegerType>()) {
//       auto tmp = builder.create<mlir::AddIOp>(
//           location, current_value,
//           get_or_create_constant_integer_value(
//               1, location, current_value.getType().getIntOrFloatBitWidth()));

//       auto memref = current_value.getDefiningOp<mlir::LoadOp>().memref();
//       builder.create<mlir::StoreOp>(
//           location, tmp, memref,
//           llvm::makeArrayRef(
//               std::vector<mlir::Value>{get_or_create_constant_index_value(
//                   0, location,
//                   current_value.getType().getIntOrFloatBitWidth())}));
//     } else {
//       printErrorMessage("we can only increment integer types.");
//     }
//   } else if (type == "--") {
//     if (current_value.getType().isa<mlir::IntegerType>()) {
//       auto tmp = builder.create<mlir::SubIOp>(
//           location, current_value,
//           get_or_create_constant_integer_value(
//               1, location, current_value.getType().getIntOrFloatBitWidth()));

//       auto memref = current_value.getDefiningOp<mlir::LoadOp>().memref();
//       builder.create<mlir::StoreOp>(
//           location, tmp, memref,
//           llvm::makeArrayRef(
//               std::vector<mlir::Value>{get_or_create_constant_index_value(
//                   0, location,
//                   current_value.getType().getIntOrFloatBitWidth())}));
//     } else {
//       printErrorMessage("we can only decrement integer types.");
//     }
//   }
//   return 0;
// }

antlrcpp::Any qasm3_expression_generator::visitAdditiveExpression(
    qasm3Parser::AdditiveExpressionContext* ctx) {
  auto location = get_location(builder, file_name, ctx);
  if (auto has_sub_additive_expr = ctx->additiveExpression()) {
    auto bin_op = ctx->binary_op->getText();

    visitChildren(has_sub_additive_expr);
    auto lhs = current_value;

    visitChildren(ctx->multiplicativeExpression());
    auto rhs = current_value;

    if (lhs.getType().isa<mlir::MemRefType>()) {
      lhs = builder.create<mlir::LoadOp>(
          location, lhs,
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
    }

    if (rhs.getType().isa<mlir::MemRefType>()) {
      rhs = builder.create<mlir::LoadOp>(
          location, rhs,
          get_or_create_constant_index_value(0, location, 64, symbol_table,
                                             builder));
    }

    if (bin_op == "+") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          if (auto op = lhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            lhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(rhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast lhs to float, but it is not constant.");
          }
        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          if (auto op = rhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            rhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(lhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast rhs to float, but it is not constant.");
          }
        } else {
          printErrorMessage("Could not perform addition, incompatible types: " +
                            ctx->getText());
        }

        createOp<mlir::AddFOp>(location, lhs, rhs);
      } else if (lhs.getType().isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        createOp<mlir::AddIOp>(location, lhs, rhs).result();
      } else {
        printErrorMessage(
            "Could not perform addition, incompatible types: " + ctx->getText(),
            {lhs, rhs});
      }
    } else if (bin_op == "-") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          if (auto op = lhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            lhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(rhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast lhs to float, but it is not constant.");
          }
        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          if (auto op = rhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            rhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(lhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast rhs to float, but it is not constant.");
          }
        } else {
          printErrorMessage(
              "Could not perform subtraction, incompatible types: " +
              ctx->getText());
        }

        createOp<mlir::SubFOp>(location, lhs, rhs);
      } else if (lhs.getType().isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        createOp<mlir::SubIOp>(location, lhs, rhs).result();
      } else {
        printErrorMessage(
            "Could not perform subtraction, incompatible types: " +
                ctx->getText(),
            {lhs, rhs});
      }
    }
    return 0;
  }

  return visitChildren(ctx);
}

antlrcpp::Any qasm3_expression_generator::visitMultiplicativeExpression(
    qasm3Parser::MultiplicativeExpressionContext* ctx) {
  auto location = get_location(builder, file_name, ctx);
  if (auto mult_expr = ctx->multiplicativeExpression()) {
    auto bin_op = ctx->binary_op->getText();

    visitExpressionTerminator(mult_expr->expressionTerminator());
    auto lhs = current_value;

    visitExpressionTerminator(ctx->expressionTerminator());
    auto rhs = current_value;

    if (bin_op == "*") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          if (auto op = lhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            lhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(rhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast lhs to float, but it is not constant.");
          }
        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          if (auto op = rhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            rhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(lhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast rhs to float, but it is not constant.");
          }
        } else {
          printErrorMessage(
              "Could not perform multiplication, incompatible types: " +
              ctx->getText());
        }

        createOp<mlir::MulFOp>(location, lhs, rhs);
      } else if (lhs.getType().isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        createOp<mlir::MulIOp>(location, lhs, rhs).result();
      } else {
        printErrorMessage(
            "Could not perform multiplication, incompatible types: " +
                ctx->getText(),
            {lhs, rhs});
      }
    } else if (bin_op == "/") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          if (auto op = lhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            lhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(rhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast lhs to float, but it is not constant.");
          }
        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          if (auto op = rhs.getDefiningOp<mlir::ConstantOp>()) {
            auto value = op.getValue()
                             .cast<mlir::IntegerAttr>()
                             .getValue()
                             .getLimitedValue();
            rhs = builder.create<mlir::ConstantOp>(
                location, mlir::FloatAttr::get(lhs.getType(), (double)value));
          } else {
            printErrorMessage(
                "Must cast rhs to float, but it is not constant.");
          }
        } else {
          printErrorMessage("Could not perform division, incompatible types: " +
                            ctx->getText());
        }

        createOp<mlir::DivFOp>(location, lhs, rhs);
      } else if (lhs.getType().isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        createOp<mlir::UnsignedDivIOp>(location, lhs, rhs).result();
      } else {
        printErrorMessage(
            "Could not perform division, incompatible types: " + ctx->getText(),
            {lhs, rhs});
      }
    }
    return 0;
  }
  return visitChildren(ctx);
}

// expressionTerminator
//     : Constant
//     | Integer
//     | RealNumber
//     | Identifier
//     | StringLiteral
//     | builtInCall
//     | kernelCall
//     | subroutineCall
//     | timingTerminator
//     | MINUS expressionTerminator
//     | LPAREN expression RPAREN
//     | expressionTerminator LBRACKET expression RBRACKET
//     | expressionTerminator incrementor
//     ;
antlrcpp::Any qasm3_expression_generator::visitExpressionTerminator(
    qasm3Parser::ExpressionTerminatorContext* ctx) {
  auto location = get_location(builder, file_name, ctx);

  // std::cout << "Analyze Expression Terminator: " << ctx->getText() << "\n";

  if (ctx->Constant()) {
    auto const_str = ctx->Constant()->getText();
    // std::cout << ctx->Constant()->getText() << "\n";
    double multiplier = ctx->MINUS() ? -1 : 1;
    double constant_val = 0.0;
    if (const_str == "pi") {
      constant_val = pi;
    } else {
      printErrorMessage("Constant " + const_str + " not implemented yet.");
    }
    auto value = multiplier * constant_val;
    auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
    createOp<mlir::ConstantOp>(location, float_attr);
    return 0;
  } else if (auto integer = ctx->Integer()) {
    // check minus
    int multiplier = ctx->MINUS() ? -1 : 1;
    auto idx = std::stoi(integer->getText());
    // std::cout << "Integer Terminator " << integer->getText() << ", " << idx
    //           << ", " << number_width << "\n";
    current_value = get_or_create_constant_integer_value(
        multiplier * idx, location,
        (internal_value_type.dyn_cast_or_null<mlir::IntegerType>()
             ? internal_value_type.cast<mlir::IntegerType>()
             : builder.getI64Type()),
        symbol_table, builder);
    return 0;
  } else if (auto real = ctx->RealNumber()) {
    // check minus
    double multiplier = ctx->MINUS() ? -1 : 1;
    auto value = multiplier * std::stod(real->getText());
    auto float_attr = mlir::FloatAttr::get(
        (internal_value_type.dyn_cast_or_null<mlir::FloatType>()
             ? internal_value_type.cast<mlir::FloatType>()
             : builder.getF64Type()),
        value);
    createOp<mlir::ConstantOp>(location, float_attr);
    return 0;
  } else if (auto id = ctx->Identifier()) {
    // std::cout << "Getting reference to variable " << id->getText() << "\n";
    mlir::Value value;
    if (id->getText() == "True") {
      value = get_or_create_constant_integer_value(
          1, location, builder.getIntegerType(1), symbol_table, builder);
    } else if (id->getText() == "False") {
      value = get_or_create_constant_integer_value(
          0, location, builder.getIntegerType(1), symbol_table, builder);
    } else {
      value = symbol_table.get_symbol(id->getText());
    }
    update_current_value(value);

    return 0;
  } else if (ctx->StringLiteral()) {
    auto sl = ctx->StringLiteral()->getText();
    sl = sl.substr(1, sl.length() - 2);
    llvm::StringRef string_type_name("StringType");
    mlir::Identifier dialect =
        mlir::Identifier::get("quantum", builder.getContext());
    auto str_type =
        mlir::OpaqueType::get(builder.getContext(), dialect, string_type_name);
    auto str_attr = builder.getStringAttr(sl);

    std::hash<std::string> hasher;
    auto hash = hasher(sl);
    std::stringstream ss;
    ss << "__internal_string_literal__" << hash;
    std::string var_name = ss.str();
    auto var_name_attr = builder.getStringAttr(var_name);

    update_current_value(builder.create<mlir::quantum::CreateStringLiteralOp>(
        location, str_type, str_attr, var_name_attr));
    return 0;

  } else if (ctx->LBRACKET()) {
    // This must be a terminator LBRACKET expression RBRACKET
    visitChildren(ctx);
    return 0;
  } else if (auto builtin = ctx->builtInCall()) {
    if (auto cast = builtin->castOperator()) {
      auto no_desig_type = cast->classicalType()->noDesignatorType();
      if (no_desig_type && no_desig_type->getText() == "bool") {
        // We can cast these things to bool...
        auto expr = builtin->expressionList()->expression(0);
        // std::cout << "EXPR: " << expr->getText() << "\n";
        if (expr->getText().find("[") != std::string::npos) {
          casting_indexed_integer_to_bool = true;
        }
        visitChildren(expr);
        auto value_type = current_value.getType();
        // std::cout << "DUMP THIS:\n";
        // value_type.dump();
        // current_value.dump();
        if (auto mem_value_type =
                value_type.dyn_cast_or_null<mlir::MemRefType>()) {
          if (mem_value_type.getElementType().isIntOrIndex() &&
              mem_value_type.getRank() == 1 &&
              mem_value_type.getShape()[0] == 1) {
            // Load this memref value
            // then add a CmpIOp to compare it to 1
            // return value will be new current_value
            auto load = builder.create<mlir::LoadOp>(
                location, current_value,
                get_or_create_constant_index_value(0, location, 64,
                                                   symbol_table, builder));
            current_value = builder.create<mlir::CmpIOp>(
                location, mlir::CmpIPredicate::eq, load,
                get_or_create_constant_integer_value(
                    1, location, mem_value_type.getElementType(), symbol_table,
                    builder));
            return 0;
          } else {
            std::cout << "See what was false: " << mem_value_type.isIntOrIndex()
                      << ", " << (mem_value_type.getRank() == 1) << ", "
                      << (mem_value_type.getShape()[0] == 1) << "\n";
            printErrorMessage("We can only cast integer types to bool. (" +
                              builtin->getText() + ").");
          }
        } else {
          // This is to catch things like bool(uint[i])
          current_value = builder.create<mlir::CmpIOp>(
              location, mlir::CmpIPredicate::eq, current_value,
              get_or_create_constant_integer_value(
                  1, location, current_value.getType(), symbol_table, builder));
          return 0;
        }
      }
    }

    printErrorMessage(
        "We only support bool(int|uint|uint[i]) cast operations.");

  } else if (auto sub_call = ctx->subroutineCall()) {
    // std::cout << "ARE WE HERE: " << ctx->subroutineCall()->getText() << "\n";
    // std::cout << ctx->subroutineCall()->Identifier()->getText() << ", "
    //           << ctx->subroutineCall()->expressionList(0)->getText() << "\n";

    auto func =
        symbol_table.get_seen_function(sub_call->Identifier()->getText());

    std::vector<mlir::Value> operands;

    auto qubit_expr_list_idx = 0;
    auto expression_list = sub_call->expressionList();
    if (expression_list.size() > 1) {
      // we have parameters
      qubit_expr_list_idx = 1;

      for (auto expression : expression_list[0]->expression()) {
        std::cout << "Subcall expr: " << expression->getText() << "\n";
        // add parameter values:
        // FIXME THIS SHOULD MATCH TYPES for FUNCTION
        auto value = std::stod(expression->getText());
        auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
        mlir::Value val =
            builder.create<mlir::ConstantOp>(location, float_attr);
        operands.push_back(val);
      }
    }

    for (auto expression : expression_list[qubit_expr_list_idx]->expression()) {
      qasm3_expression_generator qubit_exp_generator(
          builder, symbol_table, file_name,
          get_custom_opaque_type("Qubit", builder.getContext()));
      qubit_exp_generator.visit(expression);

      operands.push_back(qubit_exp_generator.current_value);
    }
    auto call_op = builder.create<mlir::CallOp>(location, func,
                                                llvm::makeArrayRef(operands));
    update_current_value(call_op.getResult(0));

    return 0;
  }

  else {
    printErrorMessage("Cannot handle this expression terminator yet: " +
                      ctx->getText());
  }

  return 0;
}

}  // namespace qcor