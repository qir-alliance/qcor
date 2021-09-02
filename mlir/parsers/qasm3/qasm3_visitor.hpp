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
  qasm3_visitor(mlir::OpBuilder b, mlir::ModuleOp m, std::string &fname,
                bool enable_nisq_conditional = false)
      : builder(b), file_name(fname), m_module(m),
        enable_nisq_ifelse(enable_nisq_conditional) {
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

  // Visit the compute-action-uncompute expression
  antlrcpp::Any visitCompute_action_stmt(qasm3Parser::Compute_action_stmtContext *context) override;

  // QCOR_EXPECT_TRUE handler
  antlrcpp::Any visitQcor_test_statement(
      qasm3Parser::Qcor_test_statementContext *context) override;

  antlrcpp::Any visitPragma(qasm3Parser::PragmaContext *ctx) override {
    // Handle the #pragma { export; } directive
    // Mark the export bool flag so that the later sub-routine handler will pick it up.
    if (ctx->statement().size() == 1 && ctx->statement(0)->getText() == "export;") {
      // The handler needs to reset this flag after handling the sub-routine.
      assert(!export_subroutine_as_callable);
      export_subroutine_as_callable = true;
      return 0;
    } else {
      return visitChildren(ctx);
    }
  }
 protected:
  // Reference to the MLIR OpBuilder and ModuleOp
  // this MLIRGen task
  mlir::OpBuilder builder;
  mlir::ModuleOp m_module;
  std::string file_name = "";
  bool enable_nisq_ifelse = false;  

  // The symbol table, keeps track of current scope
  ScopedSymbolTable symbol_table;

  // Booleans used for indicating how to construct
  // return statement for subroutines
  bool subroutine_return_statment_added = false;
  bool is_return_stmt = false;
  // Flag to indicate that we should add a callable export
  // for the next subroutine. 
  bool export_subroutine_as_callable = false;
  // Keep track of expected subroutine return type
  mlir::Type current_function_return_type;

  // Reference to MLIR Quantum Opaque Types
  mlir::Type qubit_type;
  mlir::Type array_type;
  mlir::Type result_type;

  // Loop control vars for break/continue implementation with Region-based
  // Affine/SCF Ops.
  // Strategy:
  /// - A break-able for loop will have a bool (first in the pair) to control
  /// the loop body execution. i.e., bypass the whole loop if the break
  /// condition is triggered.
  /// - The second bool is the continue condition which will bypass all
  /// the remaining ops in the body.
  /// We use a stack to handle nested loops, which are all break-able.
  std::stack<std::pair<mlir::Value, mlir::Value>> loop_control_directive_bool_vars;

  // Early return loop control directive: return statement in the loop body.
  // This will escape all loops until the *FuncOp* body and return.
  // Note: MLIR validation will require ReturnOp in the **Region** of a FuncOp.
  // First value: the boolean to control the early return (if true)
  // Second value: the return value.
  std::optional<std::pair<mlir::Value, std::optional<mlir::Value>>>
      region_early_return_vars;
  // This method will add correct number of InstOps
  // based on quantum gate broadcasting
  void createInstOps_HandleBroadcast(std::string name,
                                     std::vector<mlir::Value> qbit_values,
                                     std::vector<std::string> qbit_names,
                                     std::vector<std::string> symbol_table_qbit_keys,
                                     std::vector<mlir::Value> param_values,
                                     mlir::Location location,
                                     antlr4::ParserRuleContext* context);

  // Helper to handle range-based for loop
  void createRangeBasedForLoop(qasm3Parser::LoopStatementContext *context);
  // Helper to handle set-based for loop:
  // e.g., for i in {1,4,6,7}:
  void createSetBasedForLoop(qasm3Parser::LoopStatementContext *context);
  // While loop
  void createWhileLoop(qasm3Parser::LoopStatementContext *context);
  // Insert MLIR loop break
  void insertLoopBreak(mlir::Location &location,
                       mlir::OpBuilder *optional_builder = nullptr);
  void insertLoopContinue(mlir::Location &location,
                       mlir::OpBuilder *optional_builder = nullptr);
  // Insert a conditional return.
  // Assert that the insert location is *returnable*
  // i.e., in the FuncOp region.
  void conditionalReturn(mlir::Location &location, mlir::Value cond,
                         mlir::Value returnVal,
                         mlir::OpBuilder *optional_builder = nullptr);

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
    llvm::ArrayRef<int64_t> shaperef(shape);

    auto mem_type = mlir::MemRefType::get(shaperef, type);
    mlir::Value allocation = builder.create<mlir::AllocaOp>(location, mem_type);

    return allocation;
  }

  template <class NodeType>
  bool hasChildNodeOfType(antlr4::tree::ParseTree &in_node) {
    for (auto &child_node : in_node.children) {
      if (dynamic_cast<NodeType *>(child_node) ||
          hasChildNodeOfType<NodeType>(*child_node)) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace qcor
