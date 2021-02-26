#include <iostream>
#include "Quantum/QuantumOps.h"
#include "mlir/IR/Builders.h"
#include "qasm3_utils.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "symbol_table.hpp"

#include "exprtk.hpp"

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
    printErrorMessage("Failed to evaluate cnostant integer expression - " +
                      expr_str + ". Must be a constant integer type.");
  }

  return (int64_t)ref;
}
}  // namespace qcor