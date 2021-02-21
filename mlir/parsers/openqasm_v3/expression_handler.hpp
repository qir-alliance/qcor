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

  std::size_t number_width;
  mlir::Type internal_float_type;

  ScopedSymbolTable& symbol_table;
  mlir::Value create_constant_integer_value(const std::size_t idx,
                                            mlir::Location location) {
    auto integer_attr =
        mlir::IntegerAttr::get(builder.getIntegerType(number_width), idx);

    auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
    // symbol_table.add_constant_integer(idx, ret);
    return ret;
  }
  mlir::Value get_or_extract_qubit(const std::string& qreg_name,
                                   const std::size_t idx,
                                   mlir::Location location) {
    auto key = qreg_name + std::to_string(idx);
    if (symbol_table.has_symbol(key)) {
      return symbol_table.get_symbol(key);  // global_symbol_table[key];
    } else {
      auto qubits = symbol_table.get_symbol(qreg_name)
                        .getDefiningOp<mlir::quantum::QallocOp>()
                        .qubits();
      mlir::Value pos;
      if (symbol_table.has_constant_integer(idx)) {
        pos = symbol_table.get_constant_integer(idx);
      } else {
        pos = create_constant_integer_value(idx, location);
      }
      llvm::StringRef qubit_type_name("Qubit");
      mlir::Identifier dialect =
          mlir::Identifier::get("quantum", builder.getContext());

      auto qubit_type =
          mlir::OpaqueType::get(builder.getContext(), dialect, qubit_type_name);

      // auto pos = create_constant_integer_value(idx, location);
      auto value = builder.create<mlir::quantum::ExtractQubitOp>(
          location, qubit_type, qubits, pos);
      symbol_table.add_symbol(key, value);
      return value;
    }
  }

  mlir::Value get_or_create_constant_integer_value(const std::size_t idx,
                                                   mlir::Location location,
                                                   int width = 64) {
    if (symbol_table.has_constant_integer(idx, width)) {
      return symbol_table.get_constant_integer(idx, width);
    } else {
      auto integer_attr =
          mlir::IntegerAttr::get(builder.getIntegerType(width), idx);

      auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
      symbol_table.add_constant_integer(idx, ret, width);
      return ret;
    }
  }

  mlir::Value get_or_create_constant_index_value(const std::size_t idx,
                                                 mlir::Location location,
                                                 int width = 64) {
    auto constant_int =
        get_or_create_constant_integer_value(idx, location, width);
    return builder.create<mlir::IndexCastOp>(location, constant_int,
                                             builder.getIndexType());
  }

 public:
  mlir::Value current_value;

  qasm3_expression_generator(mlir::OpBuilder b, ScopedSymbolTable& table,
                             std::string& fname, std::size_t nw = 64)
      : builder(b), file_name(fname), symbol_table(table), number_width(nw) {
    if (nw == 64) {
      internal_float_type = builder.getF64Type();
    } else if (nw == 32) {
      internal_float_type = builder.getF32Type();

    } else if (nw == 16) {
      internal_float_type = builder.getF16Type();
    } else {
      internal_float_type = builder.getF64Type();
    }
  }

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

      // FIXME Handle if (c[1] == 1)

    } else if (auto built_in_call = ctx->builtInCall()) {
      // cast operation
      //       builtInCall
      //     : ( builtInMath | castOperator ) LPAREN expressionList RPAREN
      //     ;

      // builtInMath
      //     : 'sin' | 'cos' | 'tan' | 'exp' | 'ln' | 'sqrt' | 'popcount' |
      //     'lengthof'
      //     ;

      if (auto cast_op = built_in_call->castOperator()) {
        auto identifier =
            built_in_call->expressionList()->expression(0)->getText();
        auto classical_type = cast_op->classicalType();
        if (auto single = classical_type->singleDesignatorType()) {
        } else if (auto noDesig = classical_type->noDesignatorType()) {
          auto type = noDesig->getText();
          if (type == "bool") {
            auto value = symbol_table.get_symbol(identifier);
            auto casted = builder.create<mlir::MemRefCastOp>(
                location, value, builder.getI1Type());
            current_value = casted;
            return 0;
          }
        }
      }
    } else if (auto sub_call = ctx->subroutineCall()) {
      //    : Identifier ( LPAREN expressionList? RPAREN )? expressionList
      std::cout << ctx->subroutineCall()->getText() << "\n";
      auto func =
          symbol_table.get_seen_function(sub_call->Identifier()->getText());

      std::vector<mlir::Value> operands;

      auto qubit_expr_list_idx = 0;
      auto expression_list = sub_call->expressionList();
      if (expression_list.size() > 1) {
        // we have parameters
        qubit_expr_list_idx = 1;

        for (auto expression : expression_list[0]->expression()) {
          // add parameter values:
          // FIXME THIS SHOULD MATCH TYPES for FUNCTION
          auto value = std::stod(expression->getText());
          auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
          mlir::Value val =
              builder.create<mlir::ConstantOp>(location, float_attr);
          operands.push_back(val);
        }
      }

      for (auto expression :
           expression_list[qubit_expr_list_idx]->expression()) {
        if (expression->LBRACKET()) {
          // this is a qubit indexed from an array
          auto qbit_var_name =
              expression->expression(0)->expressionTerminator()->getText();
          auto idx_str =
              expression->expression(1)->expressionTerminator()->getText();
          auto qbit =
              get_or_extract_qubit(qbit_var_name, std::stoi(idx_str), location);
          operands.push_back(qbit);
        } else {
          // this is a qubit
          auto qbit_var_name =
              expression->expressionTerminator()->Identifier()->getText();
          auto qbit = symbol_table.get_symbol(qbit_var_name);
          operands.push_back(qbit);
        }
      }

      auto call_op = builder.create<mlir::CallOp>(location, func,
                                                  llvm::makeArrayRef(operands));
      current_value = call_op.getResult(0);
      return 0;

    } else if (ctx->kernelCall()) {
      //    : Identifier LPAREN expressionList? RPAREN
    }

    return visitChildren(ctx);
  }

  antlrcpp::Any visitExpressionTerminator(
      qasm3Parser::ExpressionTerminatorContext* ctx) override {
    auto location = get_location(builder, file_name, ctx);

    // std::cout << "Analyze Expression Terminator: " << ctx->getText() <<
    // "\n";

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
      auto float_attr = mlir::FloatAttr::get(internal_float_type, value);
      current_value = builder.create<mlir::ConstantOp>(location, float_attr);
      return 0;
    } else if (auto integer = ctx->Integer()) {
      // check minus
      int multiplier = ctx->MINUS() ? -1 : 1;
      auto idx = std::stoi(integer->getText());
      auto integer_attr = mlir::IntegerAttr::get(
          builder.getIntegerType(number_width), multiplier * idx);
      current_value = builder.create<mlir::ConstantOp>(location, integer_attr);
      symbol_table.add_constant_integer(multiplier * idx, current_value,
                                        number_width);
      return 0;
    } else if (auto real = ctx->RealNumber()) {
      // check minus
      double multiplier = ctx->MINUS() ? -1 : 1;
      auto value = multiplier * std::stod(real->getText());
      auto float_attr = mlir::FloatAttr::get(internal_float_type, value);
      current_value = builder.create<mlir::ConstantOp>(location, float_attr);
      return 0;
    } else if (auto id = ctx->Identifier()) {
      // check symbol table
      if (!symbol_table.has_symbol(id->getText())) {
        printErrorMessage("invalid identifier (" + id->getText() +
                          "), not found in symbol table.");
      }

      current_value = symbol_table.get_symbol(id->getText());
      if (current_value.getType().isa<mlir::MemRefType>()) {
        // load the value and set it as current_value
        // we assume this is a variable name, so it is
        // stored as a size 1 memref
        auto tmp = get_or_create_constant_index_value(0, location);
        llvm::ArrayRef<mlir::Value> zero_index(tmp);
        current_value =
            builder.create<mlir::LoadOp>(location, current_value, zero_index);
      }

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