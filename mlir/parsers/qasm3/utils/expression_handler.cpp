/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "expression_handler.hpp"
#include "mlir/Dialect/SCF/SCF.h"

using namespace qasm3;

namespace qcor {

void qasm3_expression_generator::update_current_value(mlir::Value v) {
  last_current_value = current_value;
  current_value = v;
  return;
}

qasm3_expression_generator::qasm3_expression_generator(mlir::OpBuilder& b,
                                                       ScopedSymbolTable& table,
                                                       std::string& fname)
    : builder(b), file_name(fname), symbol_table(table) {
  internal_value_type = builder.getI64Type();
}

qasm3_expression_generator::qasm3_expression_generator(mlir::OpBuilder& b,
                                                       ScopedSymbolTable& table,
                                                       std::string& fname,
                                                       mlir::Type t)
    : builder(b),
      file_name(fname),
      symbol_table(table),
      internal_value_type(t) {}

antlrcpp::Any qasm3_expression_generator::visitTerminal(
    antlr4::tree::TerminalNode* node) {
  auto location = builder.getUnknownLoc();  //(builder, file_name, ctx);
  if (node->getSymbol()->getText() == "[") {
    // We have hit a closing on an index
    // std::cout << "TERMNODE:\n";
    indexed_variable_value = current_value;
    if (casting_indexed_integer_to_bool) {
      if (indexed_variable_value.getType().isa<mlir::MemRefType>()) {
        internal_value_type = indexed_variable_value.getType()
                                  .cast<mlir::MemRefType>()
                                  .getElementType();
      } else {
        internal_value_type = builder.getIndexType();
      }
    } else if (indexed_variable_value.getType().isa<mlir::MemRefType>()) {
      internal_value_type = builder.getIndexType();
    }

    // Extract the variable name VARNAME[IDX] by searching parent children
    auto c = node->parent->children;
    auto it = std::find(c.begin(), c.end(), node);
    auto index = std::distance(c.begin(), it) - 1;
    indexed_variable_name = node->parent->children[index]->getText();
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
      // auto bw = indexed_variable_value.getType().getIntOrFloatBitWidth();
      
      // NOTE: UnsignedShiftRightOp (std dialect) expects operands of type "signless-integer-like"
      // i.e. although it treats the operants as unsigned, they must be of type signless (int not uint). 
      // https://mlir.llvm.org/docs/Dialects/Standard/#stdshift_right_unsigned-mlirunsignedshiftrightop

      // This is the type for all variables involved in this procedure:
      mlir::IntegerType signless_integer_like_type =
          builder.getIntegerType(indexed_variable_value.getType()
                                     .cast<mlir::MemRefType>()
                                     .getElementType()
                                     .getIntOrFloatBitWidth());
      auto casted_idx =
          builder
              .create<mlir::quantum::IntegerCastOp>(
                  location, signless_integer_like_type, current_value)
              .output();
      auto load_value =
          builder.create<mlir::LoadOp>(location, indexed_variable_value);

      auto load_value_casted =
          builder
              .create<mlir::quantum::IntegerCastOp>(
                  location, signless_integer_like_type, load_value)
              .output();
      // Note: 'std.shift_right_unsigned' op requires the same type for
      // all operands and results
      assert(load_value_casted.getType() == casted_idx.getType());
      auto shift = builder.create<mlir::UnsignedShiftRightOp>(
          location, load_value_casted, casted_idx);

      auto and_value = builder.create<mlir::AndOp>(
          location, shift,
          get_or_create_constant_integer_value(
              1, location, signless_integer_like_type, symbol_table, builder));
      update_current_value(and_value.result());
      casting_indexed_integer_to_bool = false;
    } else {
      if (internal_value_type.dyn_cast_or_null<mlir::OpaqueType>() &&
          internal_value_type.cast<mlir::OpaqueType>().getTypeData().str() ==
              "Qubit") {
        if (current_value.getType().isa<mlir::MemRefType>()) {
          if (current_value.getType().cast<mlir::MemRefType>().getRank() == 0) {
            current_value =
                builder.create<mlir::LoadOp>(location, current_value);
          } else {
            printErrorMessage("Terminator ']' -> Invalid qubit array index: ",
                              current_value);
          }
        }
        if (current_value.getType().getIntOrFloatBitWidth() < 64) {
          current_value = builder.create<mlir::ZeroExtendIOp>(
              location, current_value, builder.getI64Type());
        }
        if (!current_value.getType().isa<mlir::IntegerType>()) {
          current_value = builder.create<mlir::IndexCastOp>(
              location, builder.getI64Type(), current_value);
        }
        
        // This is a qubit extract from qubit array:
        // Need to use the qreg name convention:
        // i.e. qreg%index
        if (auto constantOp =
                current_value.getDefiningOp<mlir::ConstantOp>()) {
          // If the index is a constant value:
          const auto index_val =
              constantOp.getValue().cast<mlir::IntegerAttr>().getInt();
          const std::string qreg_name = symbol_table.get_symbol_var_name(indexed_variable_value);
          // If cannot lookup, this indicates a logic error (failed to add the qreg to the symbol table).
          assert(!qreg_name.empty());
          mlir::Value extracted_qubit = get_or_extract_qubit(
              qreg_name, index_val, location, symbol_table, builder);
          update_current_value(extracted_qubit);
        } else {
          // We're getting accessing qubits at unknown indices...
          // (current_value cannot be const-eval'ed)
          const std::string qreg_name =
              symbol_table.get_symbol_var_name(indexed_variable_value);
          symbol_table.invalidate_qubit_extracts(qreg_name);

          update_current_value(builder.create<mlir::quantum::ExtractQubitOp>(
              location, get_custom_opaque_type("Qubit", builder.getContext()),
              indexed_variable_value, current_value));
        }
      } else if (indexed_variable_value.getType().isa<mlir::OpaqueType>() &&
                 indexed_variable_value.getType()
                         .cast<mlir::OpaqueType>()
                         .getTypeData()
                         .str() == "Array") {
        // This can happen when we are using the qasm3 handler.
        // We have a Array* (probably from a std::vector<double> e.g.),
        // need to extract with the current qir qrt call, but we need to
        // know the type - double for example. We store this in the symbol
        // table variable attributes.

        auto attrs =
            symbol_table.get_variable_attributes(indexed_variable_name);
        if (attrs.empty()) {
          printErrorMessage(
              "To index a non-qubit Array we have to know the type via the "
              "variable attributes in the symbol table.");
        }

        // We have at least one, we assume the first is what 
        // we want. 
        auto attribute = attrs[0];
        mlir::Type array_type;
        if (attribute == "double") {
          array_type = builder.getF64Type();
        } else {
          // Will add more later. 
          printErrorMessage("We do not support this array type yet: " +
                            attribute);
        }

        update_current_value(
            builder.create<mlir::quantum::GeneralArrayExtractOp>(
                location, array_type, indexed_variable_value,
                cast_array_index_value_if_required(
                    indexed_variable_value.getType(), current_value, location,
                    builder)));

        // unset the variable name just in case
        indexed_variable_name = "";
      } else {
        // We are loading from a variable
        llvm::ArrayRef<mlir::Value> idx(cast_array_index_value_if_required(
            indexed_variable_value.getType(), current_value, location,
            builder));
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
    internal_value_type =
        lhs.getType().isa<mlir::MemRefType>()
            ? lhs.getType().cast<mlir::MemRefType>().getElementType()
            : lhs.getType();
    visitChildren(compare->expression(1));
    auto rhs = current_value;

    // if lhs is memref of rank 1 and size 1, this is a
    // variable and we need to load its value
    auto lhs_type = lhs.getType();
    auto rhs_type = rhs.getType();
    if (auto mem_value_type = lhs_type.dyn_cast_or_null<mlir::MemRefType>()) {
      if (mem_value_type.getElementType().isIntOrIndexOrFloat() &&
          mem_value_type.getRank() ==
              0) {  // && mem_value_type.getShape()[0] == 1) {
        // Load this memref value

        lhs = builder.create<mlir::LoadOp>(location, lhs);
      }
    }

    if (auto mem_value_type = rhs_type.dyn_cast_or_null<mlir::MemRefType>()) {
      if (mem_value_type.getElementType().isIntOrIndexOrFloat() &&
          mem_value_type.getRank() ==
              0) {  // && mem_value_type.getShape()[0] == 1) {
        // Load this memref value

        rhs = builder.create<mlir::LoadOp>(location, rhs);
      }
    }

    auto op = relational_op->getText();
    if (antlr_to_mlir_predicate.count(op)) {
      // if so, get the mlir enum representing it
      auto predicate = antlr_to_mlir_predicate[op];

      auto lhs_bw = lhs.getType().getIntOrFloatBitWidth();
      auto rhs_bw = rhs.getType().getIntOrFloatBitWidth();

      if (lhs.getType().isa<mlir::IntegerType>() ||
          lhs.getType().isa<mlir::IndexType>()) {
        if (!rhs.getType().isa<mlir::IntegerType>() &&
            !rhs.getType().isa<mlir::IndexType>()) {
          printErrorMessage("for comparison " + op +
                                " lhs was an integer type, but rhs was not.",
                            compare, {lhs, rhs});
        }

        // We need the comparison to be on the same bit width
        if (lhs_bw < rhs_bw) {
          lhs = builder.create<mlir::ZeroExtendIOp>(
              location, lhs, builder.getIntegerType(rhs_bw));
        } else if (lhs_bw > rhs_bw) {
          rhs = builder.create<mlir::ZeroExtendIOp>(
              location, rhs, builder.getIntegerType(lhs_bw));
        }

        // create the binary op value
        update_current_value(
            builder.create<mlir::CmpIOp>(location, predicate, lhs, rhs));
      } else if (lhs.getType().isa<mlir::FloatType>()) {
        if (!rhs.getType().isa<mlir::FloatType>()) {
          printErrorMessage("for comparison " + op +
                                " lhs was an float type, but rhs was not.",
                            compare, {lhs, rhs});
        }
        update_current_value(builder.create<mlir::CmpFOp>(
            location, antlr_to_mlir_fpredicate[op], lhs, rhs));
      } else {
        // Sth wrong, we cannot handle this atm.
        printErrorMessage("Unhandled comparison " + op + ": ", compare,
                          {lhs, rhs});
      }
      return 0;
    } else {
      printErrorMessage("Invalid relational operation: " + op);
    }

  } else {
    // This is just if(expr)

    found_negation_unary_op = false;
    visitChildren(compare->expression(0));
    // now just compare current_value to 1
    mlir::Type current_value_type =
        current_value.getType().isa<mlir::MemRefType>()
            ? current_value.getType().cast<mlir::MemRefType>().getElementType()
            : current_value.getType();

    if (current_value.getType().isa<mlir::MemRefType>()) {
      current_value = builder.create<mlir::LoadOp>(location, current_value);
    }
    // ,
    // get_or_create_constant_index_value(0, location, 64, symbol_table,
    //                                    builder));

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
antlrcpp::Any qasm3_expression_generator::visitIncrementor(
    qasm3Parser::IncrementorContext* ctx) {
  printErrorMessage("Alex, please implement incrementor.", ctx);
  return 0;
  //   auto location = get_location(builder, file_name, ctx);

  //   auto type = ctx->getText();
  //   if (type == "++") {
  //     if (current_value.getType().isa<mlir::IntegerType>()) {
  //       auto tmp = builder.create<mlir::AddIOp>(
  //           location, current_value,
  //           get_or_create_constant_integer_value(
  //               1, location,
  //               current_value.getType().getIntOrFloatBitWidth()));

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
  //               1, location,
  //               current_value.getType().getIntOrFloatBitWidth()));

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
}

antlrcpp::Any qasm3_expression_generator::visitAdditiveExpression(
    qasm3Parser::AdditiveExpressionContext* ctx) {
  auto location = get_location(builder, file_name, ctx);
  if (auto has_sub_additive_expr = ctx->additiveExpression()) {
    auto bin_op = ctx->binary_op->getText();

    visit(has_sub_additive_expr);
    auto lhs = current_value;

    visit(ctx->multiplicativeExpression());
    auto rhs = current_value;

    if (lhs.getType().isa<mlir::MemRefType>()) {
      lhs = builder.create<mlir::LoadOp>(location, lhs);  //,
      // get_or_create_constant_index_value(0, location, 64, symbol_table,
      //  builder));
    }

    if (rhs.getType().isa<mlir::MemRefType>()) {
      rhs = builder.create<mlir::LoadOp>(location, rhs);  //,
      // get_or_create_constant_index_value(0, location, 64, symbol_table,
      //  builder));
    }

    if (bin_op == "+") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          lhs = builder.create<mlir::SIToFPOp>(location, lhs, rhs.getType());

        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          rhs = builder.create<mlir::SIToFPOp>(location, rhs, lhs.getType());
        }

        createOp<mlir::AddFOp>(location, lhs, rhs);
      } else if ((lhs.getType().isa<mlir::IntegerType>() ||
                  lhs.getType().isa<mlir::IndexType>()) &&
                 (rhs.getType().isa<mlir::IntegerType>() ||
                  rhs.getType().isa<mlir::IndexType>())) {
        if (lhs.getType().getIntOrFloatBitWidth() <
            rhs.getType().getIntOrFloatBitWidth()) {
          lhs =
              builder.create<mlir::ZeroExtendIOp>(location, lhs, rhs.getType());
        }
        if (rhs.getType().getIntOrFloatBitWidth() <
            lhs.getType().getIntOrFloatBitWidth()) {
          rhs =
              builder.create<mlir::ZeroExtendIOp>(location, rhs, lhs.getType());
        }

        if (lhs.getType() != rhs.getType()) {
          rhs = builder.create<mlir::IndexCastOp>(location, lhs.getType(), rhs);
        }
        createOp<mlir::AddIOp>(location, lhs, rhs).result();
      } else {
        printErrorMessage("Could not perform addition, incompatible types: ",
                          ctx, {lhs, rhs});
      }
    } else if (bin_op == "-") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          lhs = builder.create<mlir::SIToFPOp>(location, lhs, rhs.getType());

        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          rhs = builder.create<mlir::SIToFPOp>(location, rhs, lhs.getType());
        }

        createOp<mlir::SubFOp>(location, lhs, rhs);
      } else if ((lhs.getType().isa<mlir::IntegerType>() ||
                  lhs.getType().isa<mlir::IndexType>()) &&
                 (rhs.getType().isa<mlir::IntegerType>() ||
                  rhs.getType().isa<mlir::IndexType>())) {
        if (lhs.getType() != rhs.getType()) {
          rhs = builder.create<mlir::IndexCastOp>(location, lhs.getType(), rhs);
        }
        createOp<mlir::SubIOp>(location, lhs, rhs).result();
      } else {
        printErrorMessage("Could not perform subtraction, incompatible types: ",
                          ctx, {lhs, rhs});
      }
    }  // else if (bin_op == "^") {

    //}
    else {
      printErrorMessage(
          bin_op + " not yet supported by the qcor qasm3 compiler.", ctx);
    }
    return 0;
  }

  return visitChildren(ctx);
}
antlrcpp::Any qasm3_expression_generator::visitXOrExpression(
    qasm3Parser::XOrExpressionContext* ctx) {
  auto location = get_location(builder, file_name, ctx);
  if (ctx->xOrExpression() && ctx->bitAndExpression()) {
    visit(ctx->xOrExpression());
    auto lhs = current_value;
    mlir::Type lhs_element_type;
    if (auto lhs_type = lhs.getType().dyn_cast_or_null<mlir::MemRefType>()) {
      lhs_element_type = lhs_type.getElementType();
      if (lhs_type.getRank() != 0) {  //} && lhs_type.getShape()[0] != 1) {
        printErrorMessage(
            "Cannot handle lhs for ^ that is memref of rank != 0.", ctx, {lhs});
      }

      lhs = builder.create<mlir::LoadOp>(location, lhs);  //,
      // get_or_create_constant_index_value(0, location, 64, symbol_table,
      //  builder));
    } else {
      lhs_element_type = lhs.getType();
    }

    // If lhs is a float, we allow integer power (2.0 ^ 5),
    // to do so we have to let expr terminator treat ints as floats
    if (lhs_element_type.isa<mlir::FloatType>()) {
      builtin_math_func_treat_ints_as_float = true;
    }
    visit(ctx->bitAndExpression());
    builtin_math_func_treat_ints_as_float = false;
    auto rhs = current_value;
    mlir::Type rhs_element_type;
    if (auto rhs_type = rhs.getType().dyn_cast_or_null<mlir::MemRefType>()) {
      rhs_element_type = rhs_type.getElementType();
      if (rhs_type.getRank() != 0) {  //} && rhs_type.getShape()[0] != 1) {
        printErrorMessage(
            "Cannot handle rhs for ^ that is memref of rank != 1 .", ctx,
            {lhs, rhs});
      }

      rhs = builder.create<mlir::LoadOp>(location, rhs);
    } else {
      rhs_element_type = rhs.getType();
    }

    // handle integer raised to another integer.
    if (lhs_element_type.isa<mlir::IntegerType>()) {
      if (!rhs_element_type.isa<mlir::IntegerType>()) {
        printErrorMessage(
            "lhs is an integer type, therefore rhs must be an integer type.",
            ctx, {lhs, rhs});
      }

      // Emulate something like
      // int product = 1;
      // for (unsigned int i = 0; i < exp; ++i) {
      //   product *= base;
      // }

      // Will need to compute this via a loop
      // Upper bound = exponent (rhs)
      mlir::Value ubs_val = rhs;
      if (!ubs_val.getType().isa<mlir::IndexType>()) {
        ubs_val = builder.create<mlir::IndexCastOp>(
            location, builder.getIndexType(), ubs_val);
      }

      // Create the result memref, initialized to 1
      llvm::ArrayRef<int64_t> shaperef{};
      mlir::Value product_memref = builder.create<mlir::AllocaOp>(
          location, mlir::MemRefType::get(shaperef, lhs_element_type));
      builder.create<mlir::StoreOp>(
          location,
          builder.create<mlir::ConstantOp>(
              location, mlir::IntegerAttr::get(lhs_element_type, 1)),
          product_memref);

      // Lower bound = 0
      mlir::Value lbs_val = get_or_create_constant_index_value(
          0, location, 64, symbol_table, builder);
      mlir::Value step_val = get_or_create_constant_index_value(
          1, location, 64, symbol_table, builder);
      builder.create<mlir::scf::ForOp>(
          location, lbs_val, ubs_val, step_val, llvm::None,
          [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
              mlir::Value iv, mlir::ValueRange itrArgs) {
            mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
            // Load - Multiply - Store
            auto x =
                nestedBuilder.create<mlir::LoadOp>(nestedLoc, product_memref);
            auto mult = nestedBuilder.create<mlir::MulIOp>(nestedLoc, x, lhs);
            nestedBuilder.create<mlir::StoreOp>(nestedLoc, mult,
                                                product_memref);
            nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc);
          });
      current_value = builder.create<mlir::LoadOp>(location, product_memref);
      return 0;
    } else if (lhs_element_type.isa<mlir::FloatType>()) {
      if (!rhs_element_type.isa<mlir::FloatType>()) {
        printErrorMessage("lhs is a float type, rhs must be an float type.",
                          ctx, {lhs, rhs});
      }

      createOp<mlir::PowFOp>(location, lhs, rhs);
      return 0;
    } else {
      printErrorMessage("Cannot compile ^ expression given lhs and rhs types.",
                        ctx, {lhs, rhs});
    }
  }

  return visitChildren(ctx);
}

antlrcpp::Any qasm3_expression_generator::visitMultiplicativeExpression(
    qasm3Parser::MultiplicativeExpressionContext* ctx) {
  auto location = get_location(builder, file_name, ctx);

  if (auto mult_expr = ctx->multiplicativeExpression()) {
    auto bin_op = ctx->binary_op->getText();
    mlir::Value lhs = [&](){
      // Handle nested multiplicative expressions:
      if (mult_expr->binary_op) {
        // The nested multiplicativeExpression within this MultiplicativeExpressionContext
        // also has a binary op:
        // Need to visit this whole multiplicativeExpression:
        visitMultiplicativeExpression(mult_expr);
        return current_value;
      } else {
        // Just a terminator value:
        visitExpressionTerminator(mult_expr->expressionTerminator());
        return current_value;
      }
    }(); 
    
    // Get the RHS
    visitExpressionTerminator(ctx->expressionTerminator());
    auto rhs = current_value;

    if (lhs.getType().isa<mlir::MemRefType>()) {
      lhs = builder.create<mlir::LoadOp>(location, lhs);  //,
      // get_or_create_constant_index_value(0, location, 64, symbol_table,
      //  builder));
    }

    if (rhs.getType().isa<mlir::MemRefType>()) {
      rhs = builder.create<mlir::LoadOp>(location, rhs);  //,
      // get_or_create_constant_index_value(0, location, 64, symbol_table,
      //  builder));
    }

    if (bin_op == "*") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          lhs = builder.create<mlir::SIToFPOp>(location, lhs, rhs.getType());

        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          rhs = builder.create<mlir::SIToFPOp>(location, rhs, lhs.getType());
        }

        createOp<mlir::MulFOp>(location, lhs, rhs);
      } else if (lhs.getType().isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        createOp<mlir::MulIOp>(location, lhs, rhs).result();
      } else {
        printErrorMessage(
            "Could not perform multiplication, incompatible types: ", ctx,
            {lhs, rhs});
      }
    } else if (bin_op == "/") {
      if (lhs.getType().isa<mlir::FloatType>() ||
          rhs.getType().isa<mlir::FloatType>()) {
        // One of these at least is a float, need to have
        // both as float
        if (!lhs.getType().isa<mlir::FloatType>()) {
          lhs = builder.create<mlir::SIToFPOp>(location, lhs, rhs.getType());

        } else if (!rhs.getType().isa<mlir::FloatType>()) {
          rhs = builder.create<mlir::SIToFPOp>(location, rhs, lhs.getType());
        }
        createOp<mlir::DivFOp>(location, lhs, rhs);
      } else if (lhs.getType().isa<mlir::IntegerType>() &&
                 rhs.getType().isa<mlir::IntegerType>()) {
        
        if (lhs.getType().getIntOrFloatBitWidth() <
            rhs.getType().getIntOrFloatBitWidth()) {
          lhs =
              builder.create<mlir::ZeroExtendIOp>(location, lhs, rhs.getType());
        } else if (rhs.getType().getIntOrFloatBitWidth() <
                   lhs.getType().getIntOrFloatBitWidth()) {
          rhs =
              builder.create<mlir::ZeroExtendIOp>(location, rhs, lhs.getType());
        }
        createOp<mlir::SignedDivIOp>(location, lhs, rhs);
      } else {
        printErrorMessage("Could not perform division, incompatible types: ",
                          ctx, {lhs, rhs});
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

  int multiplier = 1;
  if (ctx->MINUS() && ctx->expressionTerminator()) {
    visit(ctx->expressionTerminator());
    if (current_value.getType().isIntOrFloat()) {
      mlir::Attribute attr;
      if (current_value.getType().isa<mlir::FloatType>()) {
        attr = mlir::FloatAttr::get(builder.getF64Type(), -1.0);
      } else {
        attr = mlir::IntegerAttr::get(builder.getI64Type(), -1);
      }
      auto const_op = builder.create<mlir::ConstantOp>(location, attr);
      if (current_value.getType().isa<mlir::FloatType>()) {
        createOp<mlir::MulFOp>(location, const_op, current_value);
      } else {
        createOp<mlir::MulIOp>(location, const_op, current_value);
      }
    }
    return 0;
  }

  if (ctx->LPAREN() && ctx->RPAREN()) {
    visit(ctx->expression());
    return 0;
  }

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
    if (builtin_math_func_treat_ints_as_float) {
      auto value = std::stod(integer->getText());
      createOp<mlir::ConstantOp>(
          location, mlir::FloatAttr::get(builder.getF64Type(), value));
    } else {
      auto idx = std::stoi(integer->getText());
      // std::cout << "Integer Terminator " << integer->getText() << ", " << idx
      // << "\n";
      const auto getSignlessIntegerType = [](mlir::OpBuilder &opBuilder,
                                             mlir::IntegerType in_intType) {
        return in_intType.isSignless()
                   ? in_intType
                   : opBuilder.getIntegerType(in_intType.getWidth());
      };
      auto integer_attr = mlir::IntegerAttr::get(
          (internal_value_type.dyn_cast_or_null<mlir::IntegerType>()
               ? getSignlessIntegerType(
                     builder, internal_value_type.cast<mlir::IntegerType>())
               : builder.getI64Type()),
          idx);

      assert(integer_attr.getType().cast<mlir::IntegerType>().isSignless());
      if (internal_value_type.dyn_cast_or_null<mlir::IntegerType>() &&
          !internal_value_type.cast<mlir::IntegerType>().isSignless()) {
        // Make sure we cast the constant value appropriately.
        // i.e. respect the signed/unsigned of the requested type.
        current_value = builder.create<mlir::quantum::IntegerCastOp>(
            location, internal_value_type.cast<mlir::IntegerType>(),
            builder.create<mlir::ConstantOp>(location, integer_attr)).output();
      } else {
        current_value =
            builder.create<mlir::ConstantOp>(location, integer_attr);
      }
    }
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
      // If we are not in global scope and this value is
      // marked const, then I want to re-create it and return
      // that, this will mimic using global constants in downstream
      // scopes
      // if (symbol_table.get_current_scope() != 0) {
      //   auto var_attrs = symbol_table.get_variable_attributes(id->getText());
      //   if (!var_attrs.empty() && std::find(var_attrs.begin(),
      //   var_attrs.end(),
      //                                       "const") != std::end(var_attrs))
      //                                       {
      //     auto constant_val =
      //     value.getDefiningOp<mlir::ConstantOp>().value(); value =
      //     builder.create<mlir::ConstantOp>(location, constant_val);
      //   }
      // }
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
              mem_value_type.getRank() == 0) {
            // Load this memref value
            // then add a CmpIOp to compare it to 1
            // return value will be new current_value
            auto load =
                builder.create<mlir::LoadOp>(location, current_value);  //,
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
      } else if (auto single_designator =
                     cast->classicalType()->singleDesignatorType()) {
        auto expr = builtin->expressionList()->expression(0);

        if (single_designator->getText() == "int") {
          auto designator = cast->classicalType()->designator();
          auto bit_width = symbol_table.evaluate_constant_integer_expression(
              designator->expression()->getText());

          visit(expr);
          auto var_to_cast = current_value;

          if (auto mem_value_type =
                  var_to_cast.getType().dyn_cast_or_null<mlir::MemRefType>()) {
            if (mem_value_type.getElementType().isIntOrIndex() &&
                mem_value_type.getElementType().getIntOrFloatBitWidth() == 1 &&
                mem_value_type.getRank() == 1) {
              // Right now we only support casting bits to integers

              // Goal here is to toggle the bits on int[bit_width] = 0
              // Here's the formula
              // j = loop var
              // x := var_to_cast[j], so bit[j], 1 or 0
              // number to manipulate, i = 0 at first
              // Loop over j and run
              // i ^= (-x ^ i) & (1 << j);
              //
              // e.g. in c++
              // #include <stdio.h>
              // int main() {
              //   int i = 0;
              //   int bits[4] = {0, 0, 0, 1};
              //   for (int j = 0; j < 4; j++) {
              //     i ^= (-bits[j] ^ i) & (1 << j);
              //   }
              //   printf("%d\n", i);
              // }
              // will print 8
              //
              // will need std.left_shift, and, and or
              // 1. allocate new integer and set to 0
              auto int_value_type = builder.getIntegerType(bit_width);
              auto init_attr = mlir::IntegerAttr::get(int_value_type, 0);

              llvm::ArrayRef<int64_t> shaperef(1);
              auto mem_type = mlir::MemRefType::get(shaperef, int_value_type);
              mlir::Value init_allocation =
                  builder.create<mlir::AllocaOp>(location, mem_type);

              // Store the value to the 0th index of this storeop
              builder.create<mlir::StoreOp>(
                  location,
                  builder.create<mlir::ConstantOp>(location, init_attr),
                  init_allocation,
                  get_or_create_constant_index_value(0, location, 64,
                                                     symbol_table, builder));

              auto tmp = get_or_create_constant_integer_value(
                  0, location, builder.getI64Type(), symbol_table, builder);
              auto tmp2 = get_or_create_constant_index_value(
                  0, location, 64, symbol_table, builder);
              llvm::ArrayRef<mlir::Value> zero_index(tmp2);

              // Create j, the loop variable
              mlir::Value loop_var_memref = builder.create<mlir::AllocaOp>(
                  location,
                  mlir::MemRefType::get(shaperef, builder.getI64Type()));
              builder.create<mlir::StoreOp>(
                  location,
                  get_or_create_constant_integer_value(
                      0, location, builder.getI64Type(), symbol_table, builder),
                  loop_var_memref,
                  get_or_create_constant_index_value(0, location, 64,
                                                     symbol_table, builder));

              // Create loop end value, and step size
              auto b_val = get_or_create_constant_integer_value(
                  bit_width, location, builder.getI64Type(), symbol_table,
                  builder);
              auto c_val = get_or_create_constant_integer_value(
                  1, location, builder.getI64Type(), symbol_table, builder);

              auto savept = builder.saveInsertionPoint();
              auto currRegion = builder.getBlock()->getParent();
              auto headerBlock =
                  builder.createBlock(currRegion, currRegion->end());
              auto bodyBlock =
                  builder.createBlock(currRegion, currRegion->end());
              auto incBlock =
                  builder.createBlock(currRegion, currRegion->end());
              mlir::Block* exitBlock =
                  builder.createBlock(currRegion, currRegion->end());
              builder.restoreInsertionPoint(savept);

              builder.create<mlir::BranchOp>(location, headerBlock);
              builder.setInsertionPointToStart(headerBlock);

              auto load = builder.create<mlir::LoadOp>(
                  location, loop_var_memref, zero_index);
              auto cmp = builder.create<mlir::CmpIOp>(
                  location, mlir::CmpIPredicate::slt, load, b_val);
              builder.create<mlir::CondBranchOp>(location, cmp, bodyBlock,
                                                 exitBlock);

              builder.setInsertionPointToStart(bodyBlock);
              // body needs to load the loop variable
              auto j_val = builder
                               .create<mlir::LoadOp>(location, loop_var_memref,
                                                     zero_index)
                               .result();

              mlir::Value j_val_as_index = builder.create<mlir::IndexCastOp>(
                                location, j_val, builder.getIndexType());

              auto load_bit_j =
                  j_val.getType().isa<mlir::IndexType>()
                      ? builder.create<mlir::LoadOp>(location, var_to_cast,
                                                     j_val)
                      : builder.create<mlir::LoadOp>(location, var_to_cast,
                                                     j_val_as_index);
              // Extend i1 to the same width as i
              auto load_j_ext = builder.create<mlir::ZeroExtendIOp>(
                  location, load_bit_j, int_value_type);

              // Negate bits[j] to get -bit[j]`
              auto neg_load_j = builder.create<mlir::SubIOp>(
                  location,
                  builder.create<mlir::ConstantOp>(location, init_attr),
                  load_j_ext);

              // load the current value of i
              auto load_i = builder.create<mlir::LoadOp>(
                  location, init_allocation,
                  get_or_create_constant_index_value(0, location, bit_width,
                                                     symbol_table, builder));

              // first = -bits[j] ^ i
              auto xored_val =
                  builder.create<mlir::XOrOp>(location, neg_load_j, load_i);

              // (1 << j)
              // create j integer index
              // auto j_val = get_or_create_constant_integer_value(
              //     j, location, int_value_type, symbol_table, builder);
              // second = (1 << j)
              j_val = builder.create<mlir::TruncateIOp>(location, j_val,
                                                        int_value_type);
              auto shift_left_val = builder.create<mlir::ShiftLeftOp>(
                  location,
                  get_or_create_constant_integer_value(
                      1, location, int_value_type, symbol_table, builder),
                  j_val);

              // (-bits[j] ^ i) & (1 << j)
              auto result = builder.create<mlir::AndOp>(location, xored_val,
                                                        shift_left_val);

              auto load_i2 = builder.create<mlir::LoadOp>(
                  location, init_allocation,
                  get_or_create_constant_index_value(0, location, bit_width,
                                                     symbol_table, builder));
              auto result_to_store =
                  builder.create<mlir::XOrOp>(location, load_i2, result);

              auto val = builder.create<mlir::StoreOp>(
                  location, result_to_store, init_allocation,
                  get_or_create_constant_index_value(0, location, 64,
                                                     symbol_table, builder));
              builder.create<mlir::BranchOp>(location, incBlock);

              builder.setInsertionPointToStart(incBlock);
              auto load_inc = builder.create<mlir::LoadOp>(
                  location, loop_var_memref, zero_index);
              auto add =
                  builder.create<mlir::AddIOp>(location, load_inc, c_val);
              
              assert(tmp2.getType().isa<mlir::IndexType>());
              builder.create<mlir::StoreOp>(
                  location, add, loop_var_memref,
                  llvm::makeArrayRef(std::vector<mlir::Value>{tmp2}));

              builder.create<mlir::BranchOp>(location, headerBlock);

              builder.setInsertionPointToStart(exitBlock);

              symbol_table.set_last_created_block(exitBlock);

              current_value = builder.create<mlir::LoadOp>(
                  location, init_allocation,
                  get_or_create_constant_index_value(0, location, 64,
                                                     symbol_table, builder));
              return 0;
            }
          }
        }
      }
    } else if (auto math_func = builtin->builtInMath()) {
      auto expr = builtin->expressionList()->expression(0);

      // Builtin math expr should assume int -> float
      builtin_math_func_treat_ints_as_float = true;
      visit(expr);
      builtin_math_func_treat_ints_as_float = false;
      auto arg = current_value;

      if (math_func->getText() == "sin") {
        createOp<mlir::SinOp>(location, arg);
      } else if (math_func->getText() == "arcsin") {
        // arcsin(x) = 2 atan( x / (1 + sqrt(1-x2)))
        auto xsquared = builder.create<mlir::MulFOp>(location, arg, arg);
        auto one = builder.create<mlir::ConstantOp>(
            location, mlir::FloatAttr::get(builder.getF64Type(), 1.0));
        auto two = builder.create<mlir::ConstantOp>(
            location, mlir::FloatAttr::get(builder.getF64Type(), 2.0));
        auto one_minus_xsquared =
            builder.create<mlir::SubFOp>(location, one, xsquared);
        auto sqrt_one_minus_xsquared =
            builder.create<mlir::SqrtOp>(location, one_minus_xsquared);
        auto one_plus_sqrt_term = builder.create<mlir::AddFOp>(
            location, one, sqrt_one_minus_xsquared);
        auto div =
            builder.create<mlir::DivFOp>(location, arg, one_plus_sqrt_term);
        auto atan = builder.create<mlir::AtanOp>(location, div);
        createOp<mlir::MulFOp>(location, two, atan);
      } else if (math_func->getText() == "cos") {
        createOp<mlir::CosOp>(location, arg);
      } else if (math_func->getText() == "arccos") {
        // arccos(x) = 2 * arctan( (sqrt(1-x2)) / (1+x) )
        auto xsquared = builder.create<mlir::MulFOp>(location, arg, arg);
        auto one = builder.create<mlir::ConstantOp>(
            location, mlir::FloatAttr::get(builder.getF64Type(), 1.0));
        auto two = builder.create<mlir::ConstantOp>(
            location, mlir::FloatAttr::get(builder.getF64Type(), 2.0));
        auto one_minus_xsquared =
            builder.create<mlir::SubFOp>(location, one, xsquared);
        auto sqrt_one_minus_xsquared =
            builder.create<mlir::SqrtOp>(location, one_minus_xsquared);
        auto one_plus_x = builder.create<mlir::AddFOp>(location, one, arg);
        auto div = builder.create<mlir::DivFOp>(
            location, sqrt_one_minus_xsquared, one_plus_x);
        auto atan =
            builder.create<mlir::AtanOp>(location, builder.getF64Type(), div);
        createOp<mlir::MulFOp>(location, two, atan);
      } else if (math_func->getText() == "tan") {
        auto sine = builder.create<mlir::SinOp>(location, arg);
        auto cosine = builder.create<mlir::CosOp>(location, arg);
        createOp<mlir::DivFOp>(location, sine, cosine);
      } else if (math_func->getText() == "arctan") {
        createOp<mlir::AtanOp>(location, arg);
      } else if (math_func->getText() == "exp") {
        createOp<mlir::ExpOp>(location, arg);
      } else if (math_func->getText() == "ln") {
        createOp<mlir::Log2Op>(location, arg);
      } else if (math_func->getText() == "sqrt") {
        createOp<mlir::SqrtOp>(location, arg);
      } else {
        printErrorMessage("Invalid math function, or we do not support it yet.",
                          ctx);
      }

      return 0;
    }

    printErrorMessage(
        "We only support bool(int|uint|uint[i]) cast operations.");

  } else if (auto sub_call = ctx->subroutineCall()) {
    auto func =
        symbol_table.get_seen_function(sub_call->Identifier()->getText());

    std::vector<mlir::Value> operands;

    auto qubit_expr_list_idx = 0;
    auto expression_list = sub_call->expressionList();
    if (expression_list.size() > 1) {
      // we have parameters
      qubit_expr_list_idx = 1;

      for (auto expression : expression_list[0]->expression()) {
        qasm3_expression_generator param_exp_generator(builder, symbol_table,
                                                       file_name);
        param_exp_generator.visit(expression);
        auto arg = param_exp_generator.current_value;
        if (arg.getType().isa<mlir::MemRefType>()) {
          auto element_type =
              arg.getType().cast<mlir::MemRefType>().getElementType();
          if (!(element_type.isa<mlir::IntegerType>() &&
                element_type.getIntOrFloatBitWidth() == 1)) {
            arg = builder.create<mlir::LoadOp>(location, arg);
          }
        }
        operands.push_back(arg);
      }

      // Here we add all global variables
    }

    for (auto expression : expression_list[qubit_expr_list_idx]->expression()) {
      qasm3_expression_generator qubit_exp_generator(
          builder, symbol_table, file_name,
          get_custom_opaque_type("Qubit", builder.getContext()));
      qubit_exp_generator.visit(expression);
      operands.push_back(qubit_exp_generator.current_value);
    }

    auto call_op =
        builder
            .create<mlir::CallOp>(location, func, llvm::makeArrayRef(operands))
            .getResult(0);
    // If RHS is a memref<1xTYPE> then lets load it first
    if (auto rhs_mem = call_op.getType().dyn_cast_or_null<mlir::MemRefType>()) {
      call_op = builder.create<mlir::LoadOp>(location, call_op);
    }
    // printErrorMessage("HELLO should we return the loaded result here?", ctx,
    // {call_op.getResult(0)});

    update_current_value(call_op);

    return 0;
  } else if (auto kernel_call = ctx->kernelCall()) {
    // kernelCall
    // : Identifier LPAREN expressionList? RPAREN
    // ;
    auto func =
        symbol_table.get_seen_function(kernel_call->Identifier()->getText());

    // func.get
    std::vector<mlir::Value> operands;
    auto expression_list = kernel_call->expressionList()->expression();

    for (auto exp : expression_list) {
      // auto exp_str = exp->getText();
      qasm3_expression_generator exp_generator(builder, symbol_table,
                                               file_name);
      exp_generator.visit(exp);

      auto arg = exp_generator.current_value;
      if (arg.getType().isa<mlir::MemRefType>()) {
        auto element_type =
            arg.getType().cast<mlir::MemRefType>().getElementType();
        if (!(element_type.isa<mlir::IntegerType>() &&
              element_type.getIntOrFloatBitWidth() == 1)) {
          arg = builder.create<mlir::LoadOp>(location, arg);
        }
      }
      operands.push_back(arg);
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
