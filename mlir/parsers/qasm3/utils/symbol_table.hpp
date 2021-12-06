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
#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <optional>

#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace qcor {
// using SymbolTable = std::map<std::string, mlir::Value>;
struct SymbolTable {
  std::map<std::string, mlir::Value>::iterator begin() {
    return var_name_to_value.begin();
  }
  std::map<std::string, mlir::Value>::iterator end() {
    return var_name_to_value.end();
  }

  // Check if we have this symbol:
  // If this is a *root* (master) symbol, backed by a mlir::Value
  // or this is an alias (by reference), normally only for Qubits (SSA values)
  bool has_symbol(const std::string &var_name) {
    if (var_name_to_value.find(var_name) != var_name_to_value.end()) {
      return true;
    }
    const auto alias_name_check_iter =
        ref_var_name_to_orig_var_name.find(var_name);
    if (alias_name_check_iter != ref_var_name_to_orig_var_name.end()) {
      const std::string &original_var_name = alias_name_check_iter->second;
      return var_name_to_value.find(original_var_name) !=
             var_name_to_value.end();
    }
    return false;
  }

  // Add a reference alias, i.e. the two variable names are bound
  // to a single mlir::Value.
  // Note: chaining of aliasing is traced to the root var name:
  // e.g. we can support a, b (refers to a), then c refers to b.
  void add_alias(const std::string &orig_var_name,
                 const std::string &alias_var_name) {
    if (ref_var_name_to_orig_var_name.find(orig_var_name) !=
        ref_var_name_to_orig_var_name.end()) {
      // The original var name is an alias itself...
      const std::string &root_var_name =
          ref_var_name_to_orig_var_name[orig_var_name];
      ref_var_name_to_orig_var_name[alias_var_name] = root_var_name;
    } else {
      assert(var_name_to_value.find(orig_var_name) != var_name_to_value.end());
      ref_var_name_to_orig_var_name[alias_var_name] = orig_var_name;
    }
  }

  // Get the symbol (mlir::Value) taking into account potential alias chaining.
  mlir::Value get_symbol(const std::string &var_name) {
    auto iter = var_name_to_value.find(var_name);
    if (iter != var_name_to_value.end()) {
      return iter->second;
    }

    auto alias_iter = ref_var_name_to_orig_var_name.find(var_name);
    if (alias_iter != ref_var_name_to_orig_var_name.end()) {
      const std::string &root_var_name = alias_iter->second;
      assert(var_name_to_value.find(root_var_name) != var_name_to_value.end());
      return var_name_to_value[root_var_name];
    }
    printErrorMessage("Unknown symbol '" + var_name + "'.");
    return mlir::Value();
  }

  void add_or_update_symbol(const std::string &var_name, mlir::Value value) {
    var_name_to_value[var_name] = value;
  }

  void erase_symbol(const std::string &var_name) {
    auto iter = ref_var_name_to_orig_var_name.find(var_name);
    if (iter != ref_var_name_to_orig_var_name.end()) {
      var_name_to_value.erase(iter->second);
    } else {
      var_name_to_value.erase(var_name);
    }
  }
  // Compatible w/ a raw map (assuming the variable is original/root)
  mlir::Value &operator[](const std::string &var_name) {
    return var_name_to_value[var_name];
  }

  mlir::Value &at(const std::string &var_name) {
    return var_name_to_value.at(var_name);
  }

  void insert(const std::pair<std::string, mlir::Value> &new_var) {
    var_name_to_value.insert(new_var);
  }

  std::map<std::string, mlir::Value>::iterator
  find(const std::string &var_name) {
    return var_name_to_value.find(var_name);
  }

  std::map<std::string, mlir::Value>::size_type
  count(const std::string &var_name) const {
    return var_name_to_value.count(var_name);
  }

private:
  std::map<std::string, mlir::Value> var_name_to_value;
  // By reference var name aliasing map:
  // track a variable name representing references to the original mlir::Value,
  // e.g. qubit aliasing from slicing.
  std::unordered_map<std::string, std::string> ref_var_name_to_orig_var_name;
};

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

  mlir::Block* last_created_block = nullptr;

  mlir::OpBuilder* builder;

  std::map<std::string, mlir::Type> global_constant_memref_types;
  std::map<std::string, double> global_constants;

  // Map Opaque Ptr of value to key in SymbolTable
  std::map<void*, std::string> replacement_helper;

  // Map Opaque Ptr of bit value to the originating Result* (from measure)
  // Note: after Measure, we perform a casting from Result* -> bool.
  // This map tracks the Result* returns by the measure op
  // so that we can trace the originating Result*.
  std::unordered_map<void *, mlir::Value> bit_var_ptr_to_meas_result_var;

public:
  template <typename T>
  T get_global_constant(const std::string variable_name) {
    if (!global_constants.count(variable_name)) {
      printErrorMessage("Invalid global constant variable name: " +
                        variable_name);
    }
    return (T)global_constants[variable_name];
  }
  void set_op_builder(mlir::OpBuilder& b) { builder = &b; }

  void evaluate_const_global(const std::string variable_name,
                             const std::string expr_str, mlir::Type type,
                             mlir::Block& module_block,
                             mlir::Location location);

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

  // Get all visible symbols at the current scope.
  // Nearer symbols take precedence over further ones (if having the same name)
  std::unordered_map<std::string, mlir::Value> get_all_visible_symbols();

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

  void replace_symbol(mlir::Value old_value, mlir::Value new_value);
  // Returns an empty string if this Value is not tracked in the symbol table.
  std::string get_symbol_var_name(mlir::Value value);

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

  // Eval a const int expression (throw if failed)
  int64_t evaluate_constant_integer_expression(const std::string expr);
  // Returns null if this expression cannot be const-eval to an integer value.
  std::optional<int64_t>
  try_evaluate_constant_integer_expression(const std::string expr);

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
    for (int i = scope; i >= 0; i--) { // nasty bug, auto instead of int...
      if (scoped_symbol_tables[i].has_symbol(variable_name)) {
        return true;
      }
    }

    return false;
  }

  void add_symbol_ref_alias(const std::string &orig_variable_name,
                            const std::string &alias_ref_variable_name) {
    // Sanity check for debug
    assert(has_symbol(orig_variable_name));
    assert(!has_symbol(alias_ref_variable_name));
    scoped_symbol_tables[current_scope].add_alias(orig_variable_name,
                                                  alias_ref_variable_name);
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
      if (scoped_symbol_tables[i].has_symbol(variable_name)) {
        return scoped_symbol_tables[i].get_symbol(variable_name);
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
        replacement_helper[value.getAsOpaquePointer()] = variable_name;
        return;
      }
    }

    scoped_symbol_tables[scope].insert({variable_name, value});
    variable_attributes.insert({variable_name, var_attributes});
    last_value_added = value;
          
    replacement_helper.insert({value.getAsOpaquePointer(), variable_name});
  }

  // get symbol at the current scope, will search parent scopes
  mlir::Value get_symbol(const std::string variable_name) {
    if (global_constant_memref_types.count(variable_name)) {
      llvm::ArrayRef<int64_t> shape{};
      return builder->create<mlir::GetGlobalMemrefOp>(
          builder->getUnknownLoc(),
          mlir::MemRefType::get(shape,
                                global_constant_memref_types[variable_name]),
          variable_name);
    }

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

  // Util to construct a symbol name for qubit within an array (qreg)
  // This is to make sure we have a consitent symbol naming convention (for SSA tracking).
  std::string array_qubit_symbol_name(const std::string &qreg_name,
                                      const std::string &index_str) {
    // Sanity check: we should have added the qreg var to the symbol table.
    assert(has_symbol(qreg_name));
    // Use '%' separator to prevent name clashes with user-defined variables
    return qreg_name + '%' + index_str;
  }
  std::string array_qubit_symbol_name(const std::string &qreg_name, int index) {
    return array_qubit_symbol_name(qreg_name, std::to_string(index));
  }

  // Invalidate all qubit symbol tracking:
  // e.g., q%1, q%2, etc.
  // This will force a re-extract afterward, i.e. disconnect the SSA chain.
  // Rationale: 
  // In cases whereby the qubit SSA use-def chain cannot be tracked reliably,
  // we need to flush the tracking, i.e., effectively adding a barrier in the use-def chain.
  // for example, 
  // - Gates (QVS Op) in a conditional block: we disconnect the SSA chain before and after the contional blocks
  // for any qubits that are involved in that block.
  // - Ambiguous qubits: e.g. q[i] (i is not known at compile time), we need to flush the entire 
  // use-def chain on register q.
  // - Function call: passing a qreg to a subroutine.
  // Notes: during optimization passes, we may be able to reconnect/reconstruct some SSA chains   
  // thanks to inlining and loop-unrolling.
  // Empty `indices` list indicates that we flush all qubits.
  void invalidate_qubit_extracts(const std::string &qreg_name,
                                 const std::vector<int> &indices = {});
  std::optional<size_t> get_qreg_size(const std::string &qreg_name);
  void erase_symbol(const std::string& var_name);


  // Checking if a qubit SSA operand has its use properly dominated in a block.
  // i.e., returns false is this value was produced by an Op in a separate region,
  // such as If or For loop.
  bool verify_qubit_ssa_dominance_property(mlir::Value qubit,
                                           mlir::Block *current_block);
  void add_measure_bit_assignment(const mlir::Value &bit_var,
                                  const mlir::Value &result_var);
  std::optional<mlir::Value> try_lookup_meas_result(const mlir::Value &bit_var);
  std::optional<mlir::Value> try_lookup_meas_result(const std::string &bit_var_name);
  ~ScopedSymbolTable() {}
};
}  // namespace qcor