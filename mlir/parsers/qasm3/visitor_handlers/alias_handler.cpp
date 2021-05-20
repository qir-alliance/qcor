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

  // Function to process the indexIdentifier block
  // We make this a function to allow re-entrance.
  const std::function<void(const std::string &,
                           decltype(context->indexIdentifier()))>
      processIdentifierDef = [this, &location, &processIdentifierDef](
                                 const std::string &in_aliasName,
                                 decltype(context->indexIdentifier())
                                     in_indexIdentifierContext) {
        // Helper to determine the qreg size
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
                printErrorMessage(
                    "Could not infer qubit[] size from block argument.");
              }
            } else {
              printErrorMessage(
                  "Could not infer qubit[] size from block argument. No size "
                  "attribute for variable in symbol table.");
            }
          }
          return nqubits;
        };
        // The RHS has an Identifier (range- or index- based slicing)
        if (in_indexIdentifierContext->Identifier()) {
          // Get the name and symbol Value of the original register.
          // We need to determine the alias array size and cache it
          // for broadcast to work on the result array.
          auto allocated_variable =
              in_indexIdentifierContext->Identifier()->getText();
          auto allocated_symbol = symbol_table.get_symbol(allocated_variable);
          // handle q[1, 3, 5] comma syntax
          if (in_indexIdentifierContext->LBRACKET()) {
            // get the comma expression, count how many elements there are
            auto expressions =
                in_indexIdentifierContext->expressionList()->expression();
            auto n_expressions = expressions.size();
            // Create a qubit array of given size
            // which keeps alias references to Qubits in the original input
            // array.
            auto str_attr = builder.getStringAttr(in_aliasName);
            auto integer_attr =
                mlir::IntegerAttr::get(builder.getI64Type(), n_expressions);
            mlir::Value alias_allocation =
                builder.create<mlir::quantum::QaliasArrayAllocOp>(
                    location, array_type, integer_attr, str_attr);
            // Add the alias register to the symbol table
            std::cout << "Adding symbol 1\n";
            symbol_table.add_symbol(in_aliasName, alias_allocation,
                                    {std::to_string(n_expressions)});
        std::cout << "made it here\n";
            auto counter = 0;
            for (auto expr : expressions) {
              // GOAL HERE IS TO ASSIGN extracted qubits from original array
              // to the correct element of the alias array
              auto idx = symbol_table.evaluate_constant_integer_expression(
                  expr->getText());
              auto dest_idx = get_or_create_constant_integer_value(
                  counter, location, builder.getI64Type(), symbol_table,
                  builder);
              auto src_idx = get_or_create_constant_integer_value(
                  idx, location, builder.getI64Type(), symbol_table, builder);
              ++counter;

              builder.create<mlir::quantum::AssignQubitOp>(
                  location, alias_allocation, dest_idx, allocated_symbol,
                  src_idx);
            }
          } else if (auto range_def =
                         in_indexIdentifierContext->rangeDefinition()) {
            // handle range definition
            const size_t n_expr = range_def->expression().size();
            // Minimum is two expressions and not more than 3
            if (n_expr < 2 || n_expr > 3) {
              printErrorMessage("Invalid array slice range.");
            }

            auto range_start_expr = range_def->expression(0);
            auto range_stop_expr = (n_expr == 2) ? range_def->expression(1)
                                                 : range_def->expression(2);

            const auto resolve_range_value =
                [&](auto *range_item_expr) -> int64_t {
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
                (n_expr == 2) ? 1
                              : resolve_range_value(range_def->expression(1));

            // Step must not be zero:
            if (range_step == 0) {
              printErrorMessage("Invalid range: step size must be non-zero.");
            }

            // std::cout << "Range: Start = " << range_start << "; Step = " <<
            // range_step << "; Stop = " << range_stop << "\n";
            auto range_start_mlir_val = get_or_create_constant_integer_value(
                range_start, location, builder.getI64Type(), symbol_table,
                builder);
            auto range_step_mlir_val = get_or_create_constant_integer_value(
                range_step, location, builder.getI64Type(), symbol_table,
                builder);
            auto range_stop_mlir_val = get_or_create_constant_integer_value(
                range_stop, location, builder.getI64Type(), symbol_table,
                builder);
            mlir::Value array_slice =
                builder.create<mlir::quantum::ArraySliceOp>(
                    location, array_type, allocated_symbol,
                    llvm::makeArrayRef(std::vector<mlir::Value>{
                        range_start_mlir_val, range_step_mlir_val,
                        range_stop_mlir_val}));

            // Determine and cache the slice size for instruction broadcasting.
            const int64_t orig_size = get_qreg_size(allocated_variable);
            const auto slice_size_calc = [](int64_t orig_size, int64_t start,
                                            int64_t step,
                                            int64_t end) -> int64_t {
              // If step > 0 and lo > hi, or step < 0 and lo < hi, the range is
              // empty. Else for step > 0, if n values are in the range, the
              // last one is lo + (n-1)*step, which must be <= hi.  Rearranging,
              // n <= (hi
              // - lo)/step + 1, so taking the floor of the RHS gives the proper
              // value.
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
            const auto new_size =
                slice_size_calc(orig_size, range_start, range_step, range_stop);
                            std::cout << "Adding symbol 2 " << in_aliasName << "\n";

            symbol_table.add_symbol(in_aliasName, array_slice,
                                    {std::to_string(new_size)});
                                    std::cout << "HI\n";
          } else {
            printErrorMessage("Could not parse the alias statement.",
                              in_indexIdentifierContext);
          }
        } else if (in_indexIdentifierContext->indexIdentifier().size() == 2) {
          // handle concatenation
          // the RHS (is an indexIdentifier) is indexIdentifier ||
          // indexIdentifier
          auto firstIdentifier = in_indexIdentifierContext->indexIdentifier(0);
          auto secondIdentifier = in_indexIdentifierContext->indexIdentifier(1);
          const auto isPureIdentifier = [](auto *indexIdentifier) {
            // This is a simple name identifier
            return !indexIdentifier->LBRACKET() &&
                   !indexIdentifier->rangeDefinition() &&
                   indexIdentifier->indexIdentifier().empty();
          };

          // STRATEGY:
          // If the indexIdentifier is *pure* (just a var name),
          // the concatenate the var directly.
          // Otherwise, create a local identifier for the nested block (could be
          // both sides) the traverse down recursively.
          const auto process_block_and_gen_var_name =
              [&](decltype(
                  firstIdentifier) in_termIdentifierNode) -> std::string {
            static int64_t temp_var_counter = 0;
            if (isPureIdentifier(in_termIdentifierNode)) {
              return in_termIdentifierNode->Identifier()->getText();
            }
            // This is a complex one:
            const std::string new_var_name =
                "__internal_concat_temp_" + std::to_string(temp_var_counter++);
            // std::cout << "Process " << new_var_name << " = "
            //           << in_termIdentifierNode->getText() << "\n";
            processIdentifierDef(new_var_name, in_termIdentifierNode);
            return new_var_name;
          };

          // Process both blocks:
          const std::string lhs_temp_var =
              process_block_and_gen_var_name(firstIdentifier);
          const std::string rhs_temp_var =
              process_block_and_gen_var_name(secondIdentifier);
          auto first_reg_symbol = symbol_table.get_symbol(lhs_temp_var);
          auto second_reg_symbol = symbol_table.get_symbol(rhs_temp_var);
          const auto first_reg_size = get_qreg_size(lhs_temp_var);
          const auto second_reg_size = get_qreg_size(rhs_temp_var);
          mlir::Value array_concat =
              builder.create<mlir::quantum::ArrayConcatOp>(
                  location, array_type, first_reg_symbol, second_reg_symbol);
          const auto new_size = first_reg_size + second_reg_size;
          // std::cout << "Concatenate " << lhs_temp_var << "[" << first_reg_size
          //           << "] with " << rhs_temp_var << "[" << second_reg_size
          //           << "] -> " << in_aliasName << "[" << new_size << "].\n";

          symbol_table.add_symbol(in_aliasName, array_concat,
                                  {std::to_string(new_size)});
        } else {
          printErrorMessage("Could not parse the alias statement.",
                            in_indexIdentifierContext);
        }
      };

  // The name of the new alias register (LHS of the alias statement)
  // This will be associated with an alias array.
  auto alias = context->Identifier()->getText();
  // Main entrance: process top-level "alias = indexIdentifier"
  // This may call itself recursively if the RHS is a nested statement.
  // e.g. alias = q[1,3,5] || q[2] || q[4:2:8]
  processIdentifierDef(alias, context->indexIdentifier());

  // If the aliasing qreg is of size 1, just returns the Qubit.
  // (rather than an array of size 1).
  assert(!symbol_table.get_variable_attributes(alias).empty());
  const size_t alias_reg_size =
      std::stoi(symbol_table.get_variable_attributes(alias)[0]);
  if (alias_reg_size == 1) {
    auto qubit_value =
        get_or_extract_qubit(alias, 0, location, symbol_table, builder);
    // Overwrite the alias with the qubit.
    symbol_table.add_symbol(alias, qubit_value, {"1"}, true);
  }

  return 0;
}
} // namespace qcor