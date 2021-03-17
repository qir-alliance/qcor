#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitAliasStatement(
    qasm3Parser::AliasStatementContext *context) {
  auto location = get_location(builder, file_name, context);

  /** Aliasing **/
  // aliasStatement
  //     : 'let' Identifier EQUALS indexIdentifier SEMICOLON
  //     ;

  // /** Register Concatenation and Slicing **/

  // indexIdentifier
  //     : Identifier rangeDefinition
  //     | Identifier ( LBRACKET expressionList RBRACKET )?
  //     | indexIdentifier '||' indexIdentifier
  //     ;

  // The name of the new alias register, pointing to previously allocated
  // register
  auto alias = context->Identifier()->getText();

  // Get the name and symbol Value of the original register.
  // We need to determine the alias array size and cache it 
  // for broadcast to work on the result array.
  auto allocated_variable = context->indexIdentifier()->Identifier()->getText();
  auto allocated_symbol = symbol_table.get_symbol(allocated_variable);
  // Determine the first qreg size
  const auto get_qreg_size = [&](const std::string &qreg_name) {
    uint64_t nqubits;
    auto qreg_value = symbol_table.get_symbol(qreg_name);
    if (auto op = qreg_value.getDefiningOp<mlir::quantum::QallocOp>()) {
      nqubits = op.size().getLimitedValue();
    } else {
      auto attributes = symbol_table.get_variable_attributes(qreg_name);
      if (!attributes.empty()) {
        try {
          nqubits = std::stoi(attributes[0]);
        } catch (...) {
          printErrorMessage("Could not infer qubit[] size from block argument.",
                            context);
        }
      } else {
        printErrorMessage(
            "Could not infer qubit[] size from block argument. No size "
            "attribute for variable in symbol table.",
            context);
      }
    }
    return nqubits;
  };

  // handle q[1, 3, 5] comma syntax
  if (context->indexIdentifier()->LBRACKET()) {
    // get the comma expression, count how many elements there are
    auto expressions =
        context->indexIdentifier()->expressionList()->expression();
    auto n_expressions = expressions.size();
    // Create a qubit array of given size
    // which keeps alias references to Qubits in the original input array.
    auto str_attr = builder.getStringAttr(alias);
    auto integer_attr =
        mlir::IntegerAttr::get(builder.getI64Type(), n_expressions);
    mlir::Value alias_allocation =
        builder.create<mlir::quantum::QaliasArrayAllocOp>(
            location, array_type, integer_attr, str_attr);
    // Add the alias register to the symbol table
    symbol_table.add_symbol(alias, alias_allocation,
                            {std::to_string(n_expressions)});

    auto counter = 0;
    for (auto expr : expressions) {
      // GOAL HERE IS TO ASSIGN extracted qubits from original array
      // to the correct element of the alias array
      auto idx =
          symbol_table.evaluate_constant_integer_expression(expr->getText());
      auto dest_idx = get_or_create_constant_integer_value(
          counter, location, builder.getI64Type(), symbol_table, builder);
      auto src_idx = get_or_create_constant_integer_value(
          idx, location, builder.getI64Type(), symbol_table, builder);
      ++counter;

      builder.create<mlir::quantum::AssignQubitOp>(
          location, alias_allocation, dest_idx, allocated_symbol, src_idx);
    }
  } else if (auto range_def = context->indexIdentifier()->rangeDefinition()) {
    // handle range definition
    const size_t n_expr = range_def->expression().size();
    // Minimum is two expressions and not more than 3
    if (n_expr < 2 || n_expr > 3) {
      printErrorMessage("Invalid array slice range.");
    }

    auto range_start_expr = range_def->expression(0);
    auto range_stop_expr =
        (n_expr == 2) ? range_def->expression(1) : range_def->expression(2);

    const auto resolve_range_value = [&](auto *range_item_expr) -> int64_t {
      const std::string range_item_str = range_item_expr->getText();
      try {
        return std::stoi(range_item_str);
      } catch (std::exception &ex) {
        return symbol_table.evaluate_constant_integer_expression(
            range_item_str);
      }
    };

    const int64_t range_start = resolve_range_value(range_start_expr);
    const int64_t range_stop = resolve_range_value(range_stop_expr);
    const int64_t range_step =
        (n_expr == 2) ? 1 : resolve_range_value(range_def->expression(1));

    // Step must not be zero:
    if (range_step == 0) {
      printErrorMessage("Invalid range: step size must be non-zero.");
    }

    // std::cout << "Range: Start = " << range_start << "; Step = " <<
    // range_step << "; Stop = " << range_stop << "\n";
    auto range_start_mlir_val = get_or_create_constant_integer_value(
        range_start, location, builder.getI64Type(), symbol_table, builder);
    auto range_step_mlir_val = get_or_create_constant_integer_value(
        range_step, location, builder.getI64Type(), symbol_table, builder);
    auto range_stop_mlir_val = get_or_create_constant_integer_value(
        range_stop, location, builder.getI64Type(), symbol_table, builder);
    mlir::Value array_slice = builder.create<mlir::quantum::ArraySliceOp>(
        location, array_type, allocated_symbol,
        llvm::makeArrayRef(std::vector<mlir::Value>{
            range_start_mlir_val, range_step_mlir_val, range_stop_mlir_val}));
    
    // Determine and cache the slice size for instruction broadcasting.
    const int64_t orig_size = get_qreg_size(allocated_variable);
    const auto slice_size_calc = [](int64_t orig_size, int64_t start,
                                    int64_t step, int64_t end) -> int64_t {
      // If step > 0 and lo > hi, or step < 0 and lo < hi, the range is empty.
      // Else for step > 0, if n values are in the range, the last one is
      // lo + (n-1)*step, which must be <= hi.  Rearranging,
      // n <= (hi - lo)/step + 1, so taking the floor of the RHS gives
      // the proper value.
      assert(step != 0);
      // Convert to positive indices (if given as negative)
      const int64_t lo = start >= 0 ? start : orig_size + start;
      const int64_t hi = end >= 0 ? end : orig_size + end;
      if (lo == hi) {
        return 1;
      }
      if (step > 0 && lo < hi) {
        return 1 + (hi - lo) / step;
      } else if (step < 0 && lo > hi) {
        return 1 + (lo - hi) / (-step);
      } else {
        return 0;
      }
    };
    const auto new_size = slice_size_calc(orig_size, range_start, range_step, range_stop);
    symbol_table.add_symbol(alias, array_slice, {std::to_string(new_size)});
  } else {
    // handle concatenation
  }
  return 0;
}
} // namespace qcor