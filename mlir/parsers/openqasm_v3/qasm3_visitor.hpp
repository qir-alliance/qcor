#pragma once
#include <regex>

#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "qasm3BaseVisitor.h"
#include "qasm3_utils.hpp"
#include "symbol_table.hpp"

using namespace qasm3;

namespace qcor {

class qasm3_visitor : public qasm3::qasm3BaseVisitor {
 public:
  ScopedSymbolTable& getScopedSymbolTable() { return symbol_table; }

  qasm3_visitor(mlir::OpBuilder b, mlir::ModuleOp m, std::string& fname)
      : builder(b), file_name(fname), m_module(m) {
    auto context = b.getContext();
    llvm::StringRef qubit_type_name("Qubit"), array_type_name("Array"),
        result_type_name("Result");
    mlir::Identifier dialect = mlir::Identifier::get("quantum", context);
    qubit_type = mlir::OpaqueType::get(context, dialect, qubit_type_name);
    array_type = mlir::OpaqueType::get(context, dialect, array_type_name);
    result_type = mlir::OpaqueType::get(context, dialect, result_type_name);
  }

  antlrcpp::Any visitQuantumDeclaration(
      qasm3Parser::QuantumDeclarationContext* context) override;

  antlrcpp::Any visitSubroutineDefinition(
      qasm3Parser::SubroutineDefinitionContext* context) override;

  antlrcpp::Any visitBranchingStatement(
      qasm3Parser::BranchingStatementContext* context) override;

  // Classical type handling
  antlrcpp::Any visitConstantDeclaration(
      qasm3Parser::ConstantDeclarationContext* context) override;
  // --- classical declaration - single, double, no, bit ---
  antlrcpp::Any visitSingleDesignatorDeclaration(
      qasm3Parser::SingleDesignatorDeclarationContext* context) override;
  antlrcpp::Any visitNoDesignatorDeclaration(
      qasm3Parser::NoDesignatorDeclarationContext* context) override;
  antlrcpp::Any visitBitDeclaration(
      qasm3Parser::BitDeclarationContext* context) override;
  // --------//

  antlrcpp::Any visitClassicalAssignment(
      qasm3Parser::ClassicalAssignmentContext* context) override;

 protected:
  mlir::OpBuilder builder;
  mlir::ModuleOp m_module;
  std::string file_name = "";

  std::size_t current_scope = 0;

  ScopedSymbolTable symbol_table;

  // std::map<std::string, mlir::Value>& global_symbol_table;
  std::map<std::string, mlir::FuncOp> seen_functions;

  bool at_global_scope = true;

  mlir::Type qubit_type;
  mlir::Type array_type;
  mlir::Type result_type;

  void update_symbol_table(const std::string& key, mlir::Value value,
                           std::vector<std::string> variable_attributes = {}) {
    if (symbol_table.has_symbol(key)) {
      printErrorMessage(key +
                            " has already been used as a variable name (it is "
                            "in the symbol table)",
                        value);
    }
    symbol_table.add_symbol(key, value, variable_attributes);
    return;
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
      // auto pos = create_constant_integer_value(idx, location);
      auto value = builder.create<mlir::quantum::ExtractQubitOp>(
          location, qubit_type, qubits, pos);
      symbol_table.add_symbol(key, value);
      return value;
    }
  }

  mlir::Value create_constant_integer_value(const std::size_t idx,
                                            mlir::Location location) {
    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), idx);
    auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
    symbol_table.add_constant_integer(idx, ret);
    return ret;
  }
};

}  // namespace qcor

/**
antlrcpp::Any visitSubroutineCall(
      qasm3Parser::SubroutineCallContext* context) override {
    auto name = context->Identifier()->getText();

    auto line = context->getStart()->getLine();
    auto col = context->getStart()->getCharPositionInLine();
    auto location =
        builder.getFileLineColLoc(builder.getIdentifier(file_name), line, col);

    if (std::find(builtins.begin(), builtins.end(), name) != builtins.end()) {
      // this is a quantum inst

      auto expr_list = context->expressionList();
      std::vector<mlir::Value> qbit_values, param_values;
      for (auto expr_list_element : expr_list) {
        // if (expr_list_element)
        for (auto qbit_expr : expr_list_element->expression()) {
          if (qbit_expr->LBRACKET() && qbit_expr->RBRACKET()) {
            auto qreg_name = qbit_expr->expression(0)->getText();
            auto qbit_idx = qbit_expr->expression(1)->getText();

            if (!symbol_table.has_symbol(qreg_name)) {
              printErrorMessage("invalid qreg name (" + qreg_name +
                                ") for instruction " + name);
            }

            auto qbit_value =
                get_or_extract_qubit(qreg_name, std::stoi(qbit_idx), location);
            qbit_values.push_back(qbit_value);
          } else if (qbit_expr->expressionTerminator()) {
            // GATE q (q is qubit q, single qubit)
            auto qreg_name =
                qbit_expr->expressionTerminator()->Identifier()->getText();

            if (!symbol_table.has_symbol(qreg_name)) {
              printErrorMessage("invalid qreg name (" + qreg_name +
                                ") for instruction " + name);
            }

            auto qbit_value = get_or_extract_qubit(qreg_name, 0, location);
            qbit_values.push_back(qbit_value);

          } else {
            auto value = std::stod(qbit_expr->getText());
            auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
            mlir::Value val =
                builder.create<mlir::ConstantOp>(location, float_attr);
            param_values.push_back(val);
          }
        }
      }

      auto str_attr = builder.getStringAttr(name);
      builder.create<mlir::quantum::InstOp>(
          location, mlir::NoneType::get(builder.getContext()), str_attr,
          llvm::makeArrayRef(qbit_values), llvm::makeArrayRef(param_values));

    } else if (seen_functions.count(name)) {
      auto expr_list = context->expressionList();
      std::vector<mlir::Value> qbit_values;
      for (auto expr_list_element : expr_list) {
        for (auto qbit_expr : expr_list_element->expression()) {
          if (qbit_expr->LBRACKET() && qbit_expr->RBRACKET()) {
            auto qreg_name = qbit_expr->expression(0)->getText();
            auto qbit_idx = qbit_expr->expression(1)->getText();

            if (!symbol_table.has_symbol(qreg_name)) {
              printErrorMessage("invalid qreg name (" + qreg_name +
                                ") for instruction " + name);
            }

            auto qbit_value =
                get_or_extract_qubit(qreg_name, std::stoi(qbit_idx), location);
            qbit_values.push_back(qbit_value);
          } else if (qbit_expr->expressionTerminator()) {
            // GATE q (q is qubit q, single qubit)
            if (qbit_expr->expressionTerminator()->RealNumber()) {
              auto value = std::stod(
                  qbit_expr->expressionTerminator()->RealNumber()->getText());
              auto float_attr =
                  mlir::FloatAttr::get(builder.getF64Type(), value);
              mlir::Value val =
                  builder.create<mlir::ConstantOp>(location, float_attr);
              qbit_values.push_back(val);

            } else if (qbit_expr->expressionTerminator()->Constant()) {
              auto constant_str =
                  qbit_expr->expressionTerminator()->Constant()->getText();
              std::cout << "WE ARE HERE ON CONSTANT\n";
              exit(1);
              // constant_str = std::regex_replace(constant_str, "pi",
              // std::to_string(-3.1415926)); auto value =
              // std::stod(constant_str); auto float_attr =
              //     mlir::FloatAttr::get(builder.getF64Type(), value);
              // mlir::Value val =
              //     builder.create<mlir::ConstantOp>(location, float_attr);
              // qbit_values.push_back(val);
            } else if (qbit_expr->expressionTerminator()->Identifier()) {
              auto qreg_name =
                  qbit_expr->expressionTerminator()->Identifier()->getText();
              auto qbit_idx = 0;

              if (!symbol_table.has_symbol(qreg_name)) {
                printErrorMessage("invalid qreg name (" + qreg_name +
                                  ") for instruction " + name);
              }

              auto qbit_value = get_or_extract_qubit(qreg_name, 0, location);
              qbit_values.push_back(qbit_value);
            }
          } else {
            std::cout << "test we are here " << qbit_expr->getText() << "\n";
          }
        }
      }

      auto func = seen_functions[name];
      builder.create<mlir::CallOp>(location, func, qbit_values);
    }
    return visitChildren(context);
  }

  antlrcpp::Any visitQuantumGateDefinition(
      qasm3Parser::QuantumGateDefinitionContext* context) override {
    // Goal, create the mlir funcop, set the builder insertion point
    // then create a new visitor to walk this sub-tree, when it gets done
    // reset the old entry block insertion point
    auto signature = context->quantumGateSignature();

    auto gate_function_name = signature->Identifier()->getText();

    std::vector<mlir::Type> argument_types;
    std::vector<std::string> arg_names;
    auto sig_str = signature->getText();
    if (sig_str.find("(") != std::string::npos) {
      // extract variable args
      sig_str =
          sig_str.substr(sig_str.find("(") + 1,
                         sig_str.find(")") - gate_function_name.length() - 1);
      auto vars = split(sig_str, ',');
      for (auto var : vars) {
        auto s = split(var, ':');
        auto type = s[0];
        auto var_name = s[1];
        arg_names.push_back(var_name);
        // FIXME pick the right type
        argument_types.push_back(mlir::FloatType::getF64(builder.getContext()));
      }
    }

    auto identifier_list = signature->identifierList();
    for (auto id_element : identifier_list->Identifier()) {
      argument_types.push_back(qubit_type);
      arg_names.push_back(id_element->getText());
    }

    auto main_block = builder.saveInsertionPoint();

    auto func_type = builder.getFunctionType(argument_types, llvm::None);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(),
                                      gate_function_name, func_type);
    mlir::FuncOp function(proto);
    auto& entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    symbol_table.enter_new_scope();

    auto arguments = entryBlock.getArguments();
    for (int i = 0; i < arguments.size(); i++) {
      symbol_table.add_symbol(arg_names[i], arguments[i]);
      // current_local_scope_symbol_table.insert({arg_names[i],
      // arguments[i]});
    }

    auto quantum_block = context->quantumBlock();

    auto ret = visitChildren(quantum_block);

    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    m_module.push_back(function);

    builder.restoreInsertionPoint(main_block);

    // current_local_scope_symbol_table.clear();
    symbol_table.exit_scope();

    seen_functions.insert({gate_function_name, function});

    return ret;
  }

  antlrcpp::Any visitQuantumInstruction(
      qasm3Parser::QuantumInstructionContext* context) override {
    auto line = context->getStart()->getLine();
    auto col = context->getStart()->getCharPositionInLine();
    auto location =
        builder.getFileLineColLoc(builder.getIdentifier(file_name), line, col);

    // quantumInstruction
    //     : quantumGateCall
    //     | quantumPhase
    //     | quantumMeasurement
    //     | quantumBarrier
    //     ;
    auto gate_call = context->quantumGateCall();
    if (gate_call) {
      std::vector<mlir::Value> inst_args;

      auto name = gate_call->quantumGateName()->getText();
      // classical params...
      if (gate_call->expressionList()) {
        auto classical_params = gate_call->expressionList();
        for (auto expression : classical_params->expression()) {
          // First case - ry(lambda), lambda from subroutine arg
          if (expression->expressionTerminator() &&
              expression->expressionTerminator()->Identifier()) {
            auto var_name =
                expression->expressionTerminator()->Identifier()->getText();
            if (symbol_table.has_symbol(var_name)) {
              inst_args.push_back(symbol_table.get_symbol(var_name));
            } else {
              printErrorMessage(
                  "this gate rotation is "
                  "not in the current scope.");
            }
          }
          // second case, ry(-2.343), constant
          else if (expression->expressionTerminator() &&
                   expression->expressionTerminator()->Constant()) {
            auto value = std::stod(
                expression->expressionTerminator()->Constant()->getText());
            auto float_attr = mlir::FloatAttr::get(builder.getF64Type(), value);
            mlir::Value val =
                builder.create<mlir::ConstantOp>(location, float_attr);
            inst_args.push_back(val);
          }
          // third case, ry(-lambda), some expresion
          // FIXME implement
        }
      }

      for (auto idx_element :
           gate_call->indexIdentifierList()->indexIdentifier()) {
        if (symbol_table.has_symbol(idx_element->getText())) {
          inst_args.push_back(symbol_table.get_symbol(idx_element->getText()));
        } else {
          // This is probably a qubit indexed from a qreg
        }
      }

      auto str_attr = builder.getStringAttr(name);

      builder.create<mlir::quantum::InstOp>(
          location, mlir::NoneType::get(builder.getContext()), str_attr,
          llvm::makeArrayRef(inst_args),
          llvm::makeArrayRef(std::vector<mlir::Value>{}));
    }

    return visitChildren(context);
  }

  // My thoughts - we borrow the xacc model of measurements and its
  // associated memory model. We hit a measure statment, and just
  // create the measure InstOps, but we save the Result values to the
  // global symbol table, maybe associated with a custom key the encodes
  // what register and index. Basically we ignore all bit declarations

  antlrcpp::Any visitQuantumMeasurementAssignment(
      qasm3Parser::QuantumMeasurementAssignmentContext* context) override {
    auto line = context->getStart()->getLine();
    auto col = context->getStart()->getCharPositionInLine();
    auto location =
        builder.getFileLineColLoc(builder.getIdentifier(file_name), line, col);
    auto str_attr = builder.getStringAttr("mz");

    if (context->EQUALS() || context->ARROW()) {
      // we have EXPR = measure EXPR
      auto indexIdentifierList = context->indexIdentifierList();
      auto measured_list = context->quantumMeasurement()->indexIdentifierList();

      // first question is - is this measuring and array or a single qubit
      // if it is an array, then measured_list->getText() should be in
      // the global symbol table
      if (symbol_table.has_symbol(measured_list->getText())) {
        auto value = symbol_table.get_symbol(measured_list->getText());
        auto op = value.getDefiningOp<mlir::quantum::QallocOp>();
        if (!op) {
          printErrorMessage("Invalid qubit register being measured: " +
                            measured_list->getText() + ".");
        }
        auto qreg_size = op.size().getLimitedValue();

        for (int i = 0; i < qreg_size; i++) {
          std::vector<mlir::Value> qubits_for_inst;
          auto qreg_name = measured_list->getText();
          auto qbit_value = get_or_extract_qubit(qreg_name, i, location);
          qubits_for_inst.push_back(qbit_value);

          auto instop = builder.create<mlir::quantum::InstOp>(
              location, result_type, str_attr,
              llvm::makeArrayRef(qubits_for_inst),
              llvm::makeArrayRef(std::vector<mlir::Value>{}));

          std::string bit_var_name = indexIdentifierList->getText();
          if (bit_var_name.find("[") != std::string::npos) {
            if (qreg_size != 1) {
              printErrorMessage(
                  "The value being measured is "
                  "not a single qubit, but a qreg.");
            }
          }

          if (qreg_size > 1) {
            bit_var_name += "_" + std::to_string(i);
          }

          // Save the measurement result to the global symbol table.
          update_symbol_table(bit_var_name, instop.bit());
        }
      } else if (measured_list->getText().find("[") != std::string::npos) {
        // we have a r = measure q[i]
        auto bit =
            measured_list->indexIdentifier(0)->expressionList()->getText();
        auto qreg_name =
            measured_list->indexIdentifier(0)->Identifier()->getText();

        auto qbit_value =
            get_or_extract_qubit(qreg_name, std::stoi(bit), location);

        auto instop = builder.create<mlir::quantum::InstOp>(
            location, result_type, str_attr,
            llvm::makeArrayRef(std::vector<mlir::Value>{qbit_value}),
            llvm::makeArrayRef(std::vector<mlir::Value>{}));

        auto bit_var_name = indexIdentifierList->getText();
        // Save the measurement result to the global symbol table.
        update_symbol_table(bit_var_name, instop.bit());
      }
    }

    return visitChildren(context);
  }
*/