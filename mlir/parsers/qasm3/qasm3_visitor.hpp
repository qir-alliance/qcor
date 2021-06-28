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

// This class provides a set of visitor methods for the
// various nodes of the auto-generated qasm3.g4 Antlr parse tree.
// It keeps track of the translation unit symbol table and the MLIR
// OpBuilder and its goal is to build up an MLIR representation of
// the qasm3 source code using the QuantumDialect and the StdDialect.
class qasm3_visitor : public qasm3::qasm3BaseVisitor {
 public:
  // Return the symbol table.
  ScopedSymbolTable* getScopedSymbolTable() { return &symbol_table; }

  // The constructor, instantiates commonly used opaque types
  qasm3_visitor(mlir::OpBuilder b, mlir::ModuleOp m, std::string& fname)
      : builder(b), file_name(fname), m_module(m) {
    auto context = b.getContext();
    llvm::StringRef qubit_type_name("Qubit"), array_type_name("Array"),
        result_type_name("Result");
    mlir::Identifier dialect = mlir::Identifier::get("quantum", context);
    qubit_type = mlir::OpaqueType::get(context, dialect, qubit_type_name);
    array_type = mlir::OpaqueType::get(context, dialect, array_type_name);
    result_type = mlir::OpaqueType::get(context, dialect, result_type_name);
    symbol_table.set_op_builder(builder);
  }

  // Visit nodes corresponding to quantum variable and gate declarations.
  // see visitor_handlers/quantum_types_handler.cpp for implementation
  antlrcpp::Any visitQuantumGateDefinition(
      qasm3Parser::QuantumGateDefinitionContext* context) override;
  antlrcpp::Any visitQuantumDeclaration(
      qasm3Parser::QuantumDeclarationContext* context) override;

  // Visit nodes corresponding to quantum gate, subroutine, and
  // kernel calls.
  // see visitor_handlers/quantum_instruction_handler.cpp
  antlrcpp::Any visitQuantumGateCall(
      qasm3Parser::QuantumGateCallContext* context) override;
  antlrcpp::Any visitSubroutineCall(
      qasm3Parser::SubroutineCallContext* context) override;
  antlrcpp::Any visitKernelCall(
      qasm3Parser::KernelCallContext* context) override;
  antlrcpp::Any visitKernelDeclaration(
      qasm3Parser::KernelDeclarationContext* context) override;

  // Visit nodes corresponding to quantum measurement and
  // measurement assignment
  // see visitor_handlers/measurement_handler.cpp
  antlrcpp::Any visitQuantumMeasurement(
      qasm3Parser::QuantumMeasurementContext* context) override;
  antlrcpp::Any visitQuantumMeasurementAssignment(
      qasm3Parser::QuantumMeasurementAssignmentContext* context) override;

  // Visit nodes corresponding to subroutine definitions
  // and corresponding return statements
  // see visitor_handlers/subroutine_handler.cpp
  antlrcpp::Any visitSubroutineDefinition(
      qasm3Parser::SubroutineDefinitionContext* context) override;
  antlrcpp::Any visitReturnStatement(
      qasm3Parser::ReturnStatementContext* context) override;

  // Visit nodes corresponding to if/else branching statements
  // see visitor_handlers/conditional_handler.cpp
  antlrcpp::Any visitBranchingStatement(
      qasm3Parser::BranchingStatementContext* context) override;

  // Visit nodes corresponding to for and while loop statements
  // see visitor_handlers/for_stmt_handler.cpp
  antlrcpp::Any visitLoopStatement(
      qasm3Parser::LoopStatementContext* context) override;
  antlrcpp::Any visitControlDirective(
      qasm3Parser::ControlDirectiveContext* context) override;

  // Visit nodes related to classical variable declrations -
  // constants, int, float, bit, etc and then assignments to
  // those variables.
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

  // Visit nodes related to aliasing, the 'let' keyword
  // see visitor_handlers/alias_handler.cpp
  antlrcpp::Any visitAliasStatement(
      qasm3Parser::AliasStatementContext* context) override;

  antlrcpp::Any visitQcor_test_statement(
      qasm3Parser::Qcor_test_statementContext* context) override {
    auto location = get_location(builder, file_name, context);

    auto boolean_expr = context->booleanExpression();
    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(boolean_expr);
    auto expr_value = exp_generator.current_value;

    // So we have a conditional result, want
    // to negate it and see if == true
    expr_value = builder.create<mlir::CmpIOp>(
        location, mlir::CmpIPredicate::ne, expr_value,
        get_or_create_constant_integer_value(1, location, builder.getI1Type(),
                                             symbol_table, builder));

    auto currRegion = builder.getBlock()->getParent();
    auto savept = builder.saveInsertionPoint();
    auto thenBlock = builder.createBlock(currRegion, currRegion->end());
    mlir::Block* exitBlock = builder.createBlock(currRegion, currRegion->end());

    // Build up the THEN Block
    builder.setInsertionPointToStart(thenBlock);

    auto sl = "QCOR Test Failure: " + context->getText() + "\n";
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

    auto string_literal = builder.create<mlir::quantum::CreateStringLiteralOp>(
        location, str_type, str_attr, var_name_attr);
    builder.create<mlir::quantum::PrintOp>(
        location, llvm::makeArrayRef(std::vector<mlir::Value>{string_literal}));

    auto integer_attr = mlir::IntegerAttr::get(builder.getI32Type(), 1);
    auto ret = builder.create<mlir::ConstantOp>(location, integer_attr);
    builder.create<mlir::ReturnOp>(location, llvm::ArrayRef<mlir::Value>(ret));

    // Restore the insertion point and create the conditional statement
    builder.restoreInsertionPoint(savept);
    builder.create<mlir::CondBranchOp>(location, expr_value, thenBlock,
                                       exitBlock);
    builder.setInsertionPointToStart(exitBlock);

    symbol_table.set_last_created_block(exitBlock);

    return 0;
  }

 protected:
  // Reference to the MLIR OpBuilder and ModuleOp
  // this MLIRGen task
  mlir::OpBuilder builder;
  mlir::ModuleOp m_module;
  std::string file_name = "";

  // We keep reference to these blocks so that
  // we can handle break/continue correctly
  mlir::Block* current_loop_exit_block;
  mlir::Block* current_loop_header_block;
  mlir::Block* current_loop_incrementor_block;

  // The symbol table, keeps track of current scope
  ScopedSymbolTable symbol_table;

  // Booleans used for indicating how to construct
  // return statement for subroutines
  bool subroutine_return_statment_added = false;
  bool is_return_stmt = false;
  // Keep track of expected subroutine return type
  mlir::Type current_function_return_type;

  // Reference to MLIR Quantum Opaque Types
  mlir::Type qubit_type;
  mlir::Type array_type;
  mlir::Type result_type;

  // This method will add correct number of InstOps
  // based on quantum gate broadcasting
  void createInstOps_HandleBroadcast(std::string name,
                                     std::vector<mlir::Value> qbit_values,
                                     std::vector<std::string> qbit_names,
                                     std::vector<std::string> symbol_table_qbit_keys,
                                     std::vector<mlir::Value> param_values,
                                     mlir::Location location,
                                     antlr4::ParserRuleContext* context);

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

    // Assert that the values to init the memref array
    // must be of the expected type.
    for (const auto &init_val : initial_values) {
      assert(init_val.getType() == type);
    }

    // Allocate
    auto allocation = allocate_1d_memory(location, shape, type);
    // and initialize
    for (int i = 0; i < initial_values.size(); i++) {
      assert(initial_indices[i].getType().isa<mlir::IndexType>());
      builder.create<mlir::StoreOp>(location, initial_values[i], allocation,
                                    initial_indices[i]);
    }
    return allocation;
  }

  // This function serves as a utility for creating a MemRef and
  // corresponding AllocOp of a given 1d shape.
  mlir::Value allocate_1d_memory(mlir::Location location, int64_t shape,
                                 mlir::Type type) {
    llvm::ArrayRef<int64_t> shaperef{shape};

    auto mem_type = mlir::MemRefType::get(shaperef, type);
    mlir::Value allocation = builder.create<mlir::AllocaOp>(location, mem_type);

    return allocation;
  }
};

}  // namespace qcor
