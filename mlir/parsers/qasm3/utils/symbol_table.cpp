#include "qasm3_utils.hpp"

#include "symbol_table.hpp"

#include <iostream>

#include "Quantum/QuantumOps.h"
#include "exprtk.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using symbol_table_t = exprtk::symbol_table<double>;
using expression_t = exprtk::expression<double>;
using parser_t = exprtk::parser<double>;

namespace qcor {
int64_t ScopedSymbolTable::evaluate_constant_integer_expression(
    const std::string expr_str) {
  auto all_constants = get_constant_integer_variables();
  std::vector<std::string> variable_names;
  std::vector<double> variable_values;
  for (auto [n, v] : all_constants) {
    variable_names.push_back(n);
    variable_values.push_back(v);
  }

  for (auto& [name, global_value] : global_constants) {
    if (global_constant_memref_types.count(name) &&
        global_constant_memref_types[name].isa<mlir::IntegerType>()) {
      variable_names.push_back(name);
      variable_values.push_back((int64_t)global_value);
    }
  }

  double ref = 0.0;

  symbol_table_t exprtk_symbol_table;
  exprtk_symbol_table.add_constants();
  for (int i = 0; i < variable_names.size(); i++) {
    exprtk_symbol_table.add_variable(variable_names[i], variable_values[i]);
  }

  expression_t expr;
  expr.register_symbol_table(exprtk_symbol_table);
  parser_t parser;
  if (parser.compile(expr_str, expr)) {
    ref = expr.value();
  } else {
    printErrorMessage("Failed to evaluate cnostant integer expression: " +
                      expr_str + ". Must be a constant integer type.");
  }

  return (int64_t)ref;
}

void ScopedSymbolTable::evaluate_const_global(const std::string variable_name,
                                              const std::string expr_str,
                                              mlir::Type type,
                                              mlir::Block& module_block,
                                              mlir::Location location) {
  // Global Const values will be created as global_memrefs
  // These can be constant ints or floats, type will tell us
  // They can be constructed from arithmetic operations on previous
  // global const values / global_memrefs. They need to
  // be stored so that future get_symbol calls will use the getGlobalMemref op.

  std::vector<std::string> variable_names;
  std::vector<double> variable_values;

  for (auto& [name, global_value] : global_constants) {
    variable_names.push_back(name);
    variable_values.push_back(global_value);
  }

  double ref = 0.0;

  symbol_table_t exprtk_symbol_table;
  exprtk_symbol_table.add_constants();
  for (int i = 0; i < variable_names.size(); i++) {
    exprtk_symbol_table.add_variable(variable_names[i], variable_values[i]);
  }

  expression_t expr;
  expr.register_symbol_table(exprtk_symbol_table);
  parser_t parser;
  if (parser.compile(expr_str, expr)) {
    ref = expr.value();
  } else {
    printErrorMessage("Failed to evaluate constant integer expression: " +
                      expr_str + ". Must be a constant integer type.");
  }

  // Now create the Global Memref Op
  llvm::ArrayRef<int64_t> shaperef{};
  mlir::DenseElementsAttr initial_attr;
  if (type.isa<mlir::IntegerType>()) {
    initial_attr = mlir::DenseElementsAttr::get(
        mlir::VectorType::get(shaperef, type), {(int64_t)ref});
  } else {
    initial_attr = mlir::DenseElementsAttr::get(
        mlir::VectorType::get(shaperef, type), {ref});
  }

  auto memref_type = mlir::MemRefType::get(shaperef, type);
  auto savept = builder->saveInsertionPoint();
  builder->setInsertionPointToStart(&module_block);
  auto x = builder->create<mlir::GlobalMemrefOp>(
      location, variable_name, builder->getStringAttr("private"),
      mlir::TypeAttr::get(memref_type), initial_attr, true);
  builder->restoreInsertionPoint(savept);

  global_constant_memref_types.insert({variable_name, type});
  global_constants.insert({variable_name, ref});

  // add_symbol(variable_name, x, {"const"});
  return;
}
}  // namespace qcor