#pragma once
#include <regex>

#include "Quantum/QuantumOps.h"
#include "expression_handler.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
    result_type = mlir::IntegerType::get(context, 1);
  }

  // see visitor_handlers/quantum_types_handler.cpp
  antlrcpp::Any visitQuantumDeclaration(
      qasm3Parser::QuantumDeclarationContext* context) override;

  // see visitor_handlers/quantum_instruction_handler.cpp
  antlrcpp::Any visitQuantumGateCall(
      qasm3Parser::QuantumGateCallContext* context) override;
  antlrcpp::Any visitSubroutineCall(
      qasm3Parser::SubroutineCallContext* context) override;
  antlrcpp::Any visitKernelCall(
      qasm3Parser::KernelCallContext* context) override;

  // see visitor_handlers/measurement_handler.cpp
  // antlrcpp::Any visitQuantumMeasurement(
  //     qasm3Parser::QuantumMeasurementContext* context) override;
  // antlrcpp::Any visitQuantumMeasurementAssignment(
  //     qasm3Parser::QuantumMeasurementAssignmentContext* context) override;

  // // see visitor_handlers/subroutine_handler.cpp
  // antlrcpp::Any visitSubroutineDefinition(
  //     qasm3Parser::SubroutineDefinitionContext* context) override;
  // antlrcpp::Any visitReturnStatement(
  //     qasm3Parser::ReturnStatementContext* context) override;

  // // see visitor_handlers/conditional_handler.cpp
  // antlrcpp::Any visitBranchingStatement(
  //     qasm3Parser::BranchingStatementContext* context) override;

  // // see visitor_handlers/for_stmt_handler.cpp
  // antlrcpp::Any visitLoopStatement(
  //     qasm3Parser::LoopStatementContext* context) override;
  // antlrcpp::Any visitControlDirective(
  //     qasm3Parser::ControlDirectiveContext* context) override;

  // see visitor_handlers/classical_types_handler.cpp
  antlrcpp::Any visitConstantDeclaration(
      qasm3Parser::ConstantDeclarationContext* context) override;
  antlrcpp::Any visitSingleDesignatorDeclaration(
      qasm3Parser::SingleDesignatorDeclarationContext* context) override;
  antlrcpp::Any visitNoDesignatorDeclaration(
      qasm3Parser::NoDesignatorDeclarationContext* context) override;
  antlrcpp::Any visitBitDeclaration(
      qasm3Parser::BitDeclarationContext* context) override;
  antlrcpp::Any visitClassicalAssignment(
      qasm3Parser::ClassicalAssignmentContext* context) override;

  // antlrcpp::Any visitExpression(
  //     qasm3Parser::ExpressionContext* context) override {
  //   if (context->incrementor()) {
  //     qasm3_expression_generator exp_generator(builder, symbol_table,
  //                                              file_name);
  //     exp_generator.visit(context);
  //     auto expr_value = exp_generator.current_value;
  //     return 0;
  //   }
  //   return visitChildren(context);
  // }
  // --------//

  // The last block added by either loop or if stmts
  mlir::Block* current_block;

 protected:
  // Reference to the MLIR OpBuilder and ModuleOp
  // this MLIRGen task
  mlir::OpBuilder builder;
  mlir::ModuleOp m_module;
  std::string file_name = "";

  std::size_t current_scope = 0;

  // The symbol table, keeps track of current scope
  ScopedSymbolTable symbol_table;

  bool at_global_scope = true;
  bool subroutine_return_statment_added = false;
  bool is_return_stmt = false;

  mlir::Type current_function_return_type;

  mlir::Type qubit_type;
  mlir::Type array_type;
  mlir::Type result_type;

  void update_symbol_table(const std::string& key, mlir::Value value,
                           std::vector<std::string> variable_attributes = {},
                           bool overwrite = false) {
    symbol_table.add_symbol(key, value, variable_attributes, overwrite);
    return;
  }

  // mlir::Value get_or_extract_qubit(const std::string& qreg_name,
  //                                  const std::size_t idx,
  //                                  mlir::Location location) {
  //   auto key = qreg_name + std::to_string(idx);
  //   if (symbol_table.has_symbol(key)) {
  //     return symbol_table.get_symbol(key);  // global_symbol_table[key];
  //   } else {
  //     auto qubits = symbol_table.get_symbol(qreg_name);
  //     // .getDefiningOp<mlir::quantum::QallocOp>()
  //     // .qubits();
  //     mlir::Value pos = get_or_create_constant_integer_value(idx, location);

  //     // auto pos = create_constant_integer_value(idx, location);
  //     auto value = builder.create<mlir::quantum::ExtractQubitOp>(
  //         location, qubit_type, qubits, pos);
  //     symbol_table.add_symbol(key, value);
  //     return value;
  //   }
  // }

  // mlir::Value get_or_create_constant_integer_value(const std::size_t idx,
  //                                                  mlir::Location location,
  //                                                  int width = 64) {
  //   if (symbol_table.has_constant_integer(idx, width)) {
  //     return symbol_table.get_constant_integer(idx, width);
  //   } else {
  //     auto integer_attr =
  //         mlir::IntegerAttr::get(builder.getIntegerType(width), idx);

  //     auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
  //     symbol_table.add_constant_integer(idx, ret, width);
  //     return ret;
  //   }
  // }

  // mlir::Value get_or_create_constant_index_value(const std::size_t idx,
  //                                                mlir::Location location,
  //                                                int width = 64) {
  //   auto constant_int =
  //       get_or_create_constant_integer_value(idx, location, width);
  //   return builder.create<mlir::IndexCastOp>(location, constant_int,
  //                                            builder.getIndexType());
  // }

  // This function serves as a utility for creating a MemRef and
  // corresponding AllocOp of a given 1d shape. It will also store
  // initial values to all elements of the 1d array.
  mlir::Value allocate_1d_memory_and_initialize(
      mlir::Location location, int64_t shape, mlir::Type type,
      std::vector<mlir::Value> initial_values,
      llvm::ArrayRef<mlir::Value> initial_indices) {
    if (shape != initial_indices.size()) {
      printErrorMessage(
          "Cannot allocate and initialize memory, shape and number of initial "
          "value indices is incorrect");
    }
    llvm::ArrayRef<int64_t> shaperef{shape};

    auto mem_type = mlir::MemRefType::get(shaperef, type);
    mlir::Value allocation = builder.create<mlir::AllocaOp>(location, mem_type);
    for (int i = 0; i < initial_values.size(); i++) {
      builder.create<mlir::StoreOp>(location, initial_values[i], allocation,
                                    initial_indices[i]);
    }
    return allocation;
  }

  mlir::Value allocate_1d_memory(mlir::Location location, int64_t shape,
                                 mlir::Type type) {
    llvm::ArrayRef<int64_t> shaperef{shape};

    auto mem_type = mlir::MemRefType::get(shaperef, type);
    mlir::Value allocation = builder.create<mlir::AllocaOp>(location, mem_type);

    return allocation;
  }
};

}  // namespace qcor
