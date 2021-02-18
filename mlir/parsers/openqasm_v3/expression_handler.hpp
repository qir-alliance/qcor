#pragma once

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "qasm3BaseVisitor.h"
#include "qasm3Parser.h"
#include "qasm3_utils.hpp"
#include "symbol_table.hpp"
static constexpr double pi = 3.141592653589793238;
using namespace qasm3;

namespace qcor {
class qasm3_expression_generator : public qasm3::qasm3BaseVisitor {
 protected:
  mlir::OpBuilder builder;
  mlir::ModuleOp m_module;
  std::string file_name = "";
  // std::map<std::string, mlir::Value>& global_symbol_table;

  ScopedSymbolTable& symbol_table;

 public:
  mlir::Value current_value;

  qasm3_expression_generator(mlir::OpBuilder b, ScopedSymbolTable& table,
                             std::string& fname)
      : builder(b), file_name(fname), symbol_table(table) {}

  antlrcpp::Any visitExpression(qasm3Parser::ExpressionContext* ctx) override {
    auto location = get_location(builder, file_name, ctx);

    if (auto binary_operator = ctx->binaryOperator()) {
      // std::cout << "Analyze Binary Operator: " << ctx->getText() << ", "
                // << binary_operator->getText() << "\n";
      visitChildren(ctx->expression(0));
      auto lhs = current_value;
      visitChildren(ctx->expression(1));
      auto rhs = current_value;

      auto bin_op_str = binary_operator->getText();

      // see if this is a comparison operator
      if (antlr_to_mlir_predicate.count(bin_op_str)) {
        // if so, get the mlir enum representing it
        auto predicate = antlr_to_mlir_predicate[bin_op_str];
        // create the binary op value
        current_value =
            builder.create<mlir::CmpIOp>(location, predicate, lhs, rhs);
      } else if (bin_op_str == "+") {
        if (lhs.getType().isa<mlir::FloatType>() ||
            rhs.getType().isa<mlir::FloatType>()) {
          current_value = builder.create<mlir::AddFOp>(location, lhs, rhs);
        } else if (lhs.getType().isa<mlir::IntegerType>() &&
                   rhs.getType().isa<mlir::IntegerType>()) {
          current_value = builder.create<mlir::AddIOp>(location, lhs, rhs);
        } else {
          printErrorMessage("Can't handle this type of addition yet.",
                            {lhs, rhs});
        }
      } else if (bin_op_str == "-") {
        if (lhs.getType().isa<mlir::FloatType>() ||
            rhs.getType().isa<mlir::FloatType>()) {
          current_value = builder.create<mlir::SubFOp>(location, lhs, rhs);
        } else if (lhs.getType().isa<mlir::IntegerType>() &&
                   rhs.getType().isa<mlir::IntegerType>()) {
          current_value =
              builder.create<mlir::SubIOp>(location, lhs, rhs).result();
        }
      } else if (bin_op_str == "*") {
        if (lhs.getType().isa<mlir::FloatType>() ||
            rhs.getType().isa<mlir::FloatType>()) {
          current_value = builder.create<mlir::MulFOp>(location, lhs, rhs);
        } else if (lhs.getType().isa<mlir::IntegerType>() &&
                   rhs.getType().isa<mlir::IntegerType>()) {
          current_value =
              builder.create<mlir::MulIOp>(location, lhs, rhs).result();
        }
      } else if (bin_op_str == "/") {
        if (lhs.getType().isa<mlir::FloatType>() ||
            rhs.getType().isa<mlir::FloatType>()) {
          current_value = builder.create<mlir::DivFOp>(location, lhs, rhs);
        } else if (lhs.getType().isa<mlir::IntegerType>() &&
                   rhs.getType().isa<mlir::IntegerType>()) {
          current_value =
              builder.create<mlir::UnsignedDivIOp>(location, lhs, rhs).result();
        }
      } else if (bin_op_str == "&&") {
        current_value = builder.create<mlir::AndOp>(location, lhs, rhs);

      } else if (bin_op_str == "||") {
        current_value = builder.create<mlir::OrOp>(location, lhs, rhs);
      } else {
        printErrorMessage("Invalid binary operator for if stmt: " +
                          binary_operator->getText());
      }
      return 0;
    } else if (ctx->unaryOperator()) {
      auto value = visitChildren(ctx->expression(0));
    } else if (ctx->LBRACKET()) {
      // this is expr [ expr ]
      //    | expression LBRACKET expression RBRACKET

    } else if (ctx->builtInCall()) {
      // cast operation
      //       builtInCall
      //     : ( builtInMath | castOperator ) LPAREN expressionList RPAREN
      //     ;

      // builtInMath
      //     : 'sin' | 'cos' | 'tan' | 'exp' | 'ln' | 'sqrt' | 'popcount' |
      //     'lengthof'
      //     ;
    } else if (ctx->subroutineCall()) {
      //    : Identifier ( LPAREN expressionList? RPAREN )? expressionList
    } else if (ctx->kernelCall()) {
      //    : Identifier LPAREN expressionList? RPAREN
    }

    return visitChildren(ctx);
  }

  antlrcpp::Any visitExpressionTerminator(
      qasm3Parser::ExpressionTerminatorContext* ctx) override {
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
      current_value = builder.create<mlir::ConstantOp>(location, float_attr);
      return 0;
    } else if (auto integer = ctx->Integer()) {
      // check minus
      int multiplier = ctx->MINUS() ? -1 : 1;
      auto idx = std::stoi(integer->getText());
      if (symbol_table.has_constant_integer(multiplier * idx)) {
        current_value = symbol_table.get_constant_integer(multiplier * idx);
      } else {
        auto integer_attr =
            mlir::IntegerAttr::get(builder.getI64Type(), multiplier * idx);
        current_value =
            builder.create<mlir::ConstantOp>(location, integer_attr);
        symbol_table.add_constant_integer(multiplier*idx, current_value);
      }

      return 0;
    } else if (auto real = ctx->RealNumber()) {
      // check minus
      double multiplier = ctx->MINUS() ? -1 : 1;
      auto value = multiplier * std::stod(real->getText());
      auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
      current_value = builder.create<mlir::ConstantOp>(location, float_attr);
      return 0;
    } else if (auto id = ctx->Identifier()) {
      // check symbol table
      if (!symbol_table.has_symbol(id->getText())) {
        printErrorMessage("invalid identifier (" + id->getText() +
                          "), not found in symbol table.");
      }

      current_value = symbol_table.get_symbol(id->getText());

    } else if (ctx->StringLiteral()) {
      printErrorMessage("StringLiteral not implemented yet.");
    } else {
      printErrorMessage("Cannot handle this expression terminator yet: " +
                        ctx->getText());
    }

    // std::cout << "made it here 2\n";
    return 0;  // visitChildren(ctx);
  }
};
}  // namespace qcor