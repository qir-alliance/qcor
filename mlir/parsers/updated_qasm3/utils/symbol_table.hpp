#pragma once

#include <map>
#include <vector>

#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace qcor {
using SymbolTable = std::map<std::string, mlir::Value>;
using ConstantIntegerTable =
    std::map<std::pair<std::uint64_t, int>, mlir::Value>;

// Rudimentary scoped symbol table that will keep track of
// created mlir::Values keyed on their unique variable name
// at the current scope.
class ScopedSymbolTable {
 protected:
  std::vector<SymbolTable> scoped_symbol_tables;
  std::size_t current_scope = 0;
  std::map<std::string, std::vector<std::string>> variable_attributes;

  // key is (int, width)
  std::vector<ConstantIntegerTable> constant_integer_values;
  // std::map<std::pair<std::uint64_t, int>, mlir::Value>
  // constant_integer_values;

  mlir::Value last_value_added;

  std::map<std::string, mlir::FuncOp> seen_functions;

  mlir::Block* last_created_block;

 public:
  ScopedSymbolTable() {
    // start the global table
    scoped_symbol_tables.push_back(SymbolTable{});
    constant_integer_values.push_back(ConstantIntegerTable{});
  }

  void print() {
    std::cout << "\n";
    for (int i = current_scope; i >= 0; i--) {
      std::cout << "Scope: " << i << "\n";
      for (auto& [k, v] : scoped_symbol_tables[i]) {
        std::cout << "  Variable: " << k << "\n";
        std::cout << "  MLIRDump:\n  ";
        v.dump();
        std::cout << "\n";
      }
    }
  }

  // Create new scope symbol table
  // will push_back on scoped_symbol_tables;
  void enter_new_scope() {
    current_scope++;
    scoped_symbol_tables.emplace_back(SymbolTable{});
    constant_integer_values.emplace_back(ConstantIntegerTable{});
  }

  // Exit scope, will remove the last
  // scope in scope_symbol_tables
  void exit_scope() {
    current_scope--;
    scoped_symbol_tables.pop_back();
    constant_integer_values.pop_back();
    // last_created_block = nullptr;
  }

  void set_last_created_block(mlir::Block* b) { last_created_block = b; }
  mlir::Block* get_last_created_block() { return last_created_block; }

  void add_seen_function(const std::string name, mlir::FuncOp function) {
    seen_functions.insert({name, function});
    return;
  }

  std::vector<std::string> get_seen_function_names() {
    std::vector<std::string> fnames;
    for (auto [name, value] : seen_functions) {
      fnames.push_back(name);
    }
    return fnames;
  }
  std::map<std::string, int64_t> get_constant_integer_variables() {
    std::map<std::string, int64_t> ret;
    for (int i = current_scope; i >= 0; i--) {
      auto constant_ops = get_symbols_of_type_at_scope<mlir::ConstantOp>(i);
      for (auto [var_name, op] : constant_ops) {
        if (op.getValue().isa<mlir::IntegerAttr>()) {
          ret.insert(
              {var_name, op.getValue().cast<mlir::IntegerAttr>().getInt()});
        }
      }
    }
    return ret;
  }

  int64_t evaluate_constant_integer_expression(const std::string expr);

  mlir::FuncOp get_seen_function(const std::string name) {
    if (!seen_functions.count(name)) {
      printErrorMessage(name + " is not a known function in the symbol table.");
    }
    return seen_functions[name];
  }

  bool has_seen_function(const std::string name) {
    return seen_functions.count(name);
  }

  bool has_constant_integer(std::uint64_t key, int width = 64) {
    return constant_integer_values[current_scope].count({key, width});
  }

  void add_constant_integer(std::uint64_t key, mlir::Value constant_value,
                            int width = 64) {
    if (!constant_integer_values[current_scope].count({key, width})) {
      constant_integer_values[current_scope].insert(
          {{key, width}, constant_value});
    }
    return;
  }
  mlir::Value get_constant_integer(std::uint64_t key, int width = 64) {
    if (!has_constant_integer(key, width)) {
      printErrorMessage("constant integer " + std::to_string(key) +
                        " is not in the symbol table.");
    }
    return constant_integer_values[current_scope][{key, width}];
  }
  bool is_allocation(const std::string variable_name) {
    return has_symbol(variable_name) &&
           get_symbol(variable_name).getDefiningOp<mlir::AllocOp>();
  }

  std::vector<std::string> get_variable_attributes(
      const std::string variable_name) {
    if (!has_symbol(variable_name)) {
      printErrorMessage("Variable " + variable_name +
                        " does not have any attributes.");
    }
    return variable_attributes[variable_name];
  }

  bool is_variable_mutable(const std::string variable_name) {
    if (!has_symbol(variable_name)) {
      printErrorMessage("Cannot check variable mutability, variable " +
                        variable_name + " does not exist.");
    }
    auto attrs = variable_attributes[variable_name];
    return std::find(attrs.begin(), attrs.end(), "const") == std::end(attrs);
  }

  bool has_symbol(const std::string variable_name) {
    return has_symbol(variable_name, current_scope);
  }

  bool has_symbol(const std::string variable_name, const std::size_t scope) {
    for (int i = scope; i >= 0; i--) {  // nasty bug, auto instead of int...
      if (!scoped_symbol_tables[i].empty() &&
          scoped_symbol_tables[i].count(variable_name)) {
        return true;
      }
    }

    return false;
  }

  SymbolTable& get_global_symbol_table() { return scoped_symbol_tables[0]; }

  template <typename OpTy>
  std::vector<OpTy> get_global_symbols_of_type() {
    std::vector<OpTy> ret;
    for (auto& [var_name, value] : get_global_symbol_table()) {
      auto op = value.template getDefiningOp<OpTy>();
      if (op) {
        ret.push_back(op);
      }
    }
    return ret;
  }

  template <typename OpTy>
  std::vector<std::pair<std::string, OpTy>> get_symbols_of_type_at_scope(
      const std::size_t scope) {
    std::vector<std::pair<std::string, OpTy>> ret;
    for (auto& [var_name, value] : scoped_symbol_tables[scope]) {
      auto op = value.template getDefiningOp<OpTy>();
      if (op) {
        ret.push_back({var_name, op});
      }
    }
    return ret;
  }

  // retrieve the symbol at the given scope, will search parent scopes
  mlir::Value get_symbol(const std::string variable_name,
                         const std::size_t scope) {
    for (auto i = scope; i >= 0; i--) {
      if (scoped_symbol_tables[i].count(variable_name)) {
        return scoped_symbol_tables[i][variable_name];
      }
    }

    printErrorMessage("No variable " + variable_name +
                      " in scoped symbol table (provided scope = " +
                      std::to_string(scope) + "). Did you allocate it?");
  }

  mlir::Value get_last_value_added() { return last_value_added; }

  // add the symbol to the given scope
  void add_symbol(const std::string variable_name, mlir::Value value,
                  const std::size_t scope,
                  std::vector<std::string> var_attributes = {},
                  bool overwrite = false) {
    if (scope > current_scope) {
      printErrorMessage("Provided scope is greater than the current scope.\n");
    }

    if (has_symbol(variable_name)) {
      if (!overwrite) {
        printErrorMessage(variable_name + " is already in the symbol table.");
      } else {
        scoped_symbol_tables[scope][variable_name] = value;
        variable_attributes[variable_name] = var_attributes;
        last_value_added = value;
        return;
      }
    }

    scoped_symbol_tables[scope].insert({variable_name, value});
    variable_attributes.insert({variable_name, var_attributes});
    last_value_added = value;
  }

  // get symbol at the current scope, will search parent scopes
  mlir::Value get_symbol(const std::string variable_name) {
    if (!has_symbol(variable_name)) {
      printErrorMessage("invalid symbol, not in the symbol table - " +
                        variable_name);
    }
    return get_symbol(variable_name, current_scope);
  }

  // add symbol to current scope
  void add_symbol(const std::string variable_name, mlir::Value value,
                  std::vector<std::string> var_attributes = {},
                  bool overwrite = false) {
    add_symbol(variable_name, value, current_scope, var_attributes, overwrite);
  }

  // add symbol to global scope
  void add_global_symbol(const std::string variable_name, mlir::Value value) {
    add_symbol(variable_name, value, 0);
  }

  // return the current scope
  std::size_t get_current_scope() { return current_scope; }

  // return the parent scope
  std::size_t get_parent_scope() {
    return current_scope >= 1 ? current_scope - 1 : 0;
  }
  ~ScopedSymbolTable() {}
};
}  // namespace qcor