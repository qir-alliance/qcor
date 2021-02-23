#include "expression_handler.hpp"
#include "exprtk.hpp"
#include "qasm3_visitor.hpp"
using symbol_table_t = exprtk::symbol_table<double>;
using expression_t = exprtk::expression<double>;
using parser_t = exprtk::parser<double>;

namespace qcor {
antlrcpp::Any qasm3_visitor::visitLoopStatement(
    qasm3Parser::LoopStatementContext* context) {
  auto location = get_location(builder, file_name, context);

  auto loop_signature = context->loopSignature();
  auto program_block = context->programBlock();

  //   std::cout << loop_signature->getText() << "\n";
  //   std::cout << program_block->getText() << "\n";

  if (auto membership_test = loop_signature->membershipTest()) {
    // this is a for loop
    auto idx_var_name = membership_test->Identifier()->getText();

    auto set_declaration = membership_test->setDeclaration();
    if (set_declaration->LBRACE()) {
      auto exp_list = set_declaration->expressionList();
      auto n_expr = exp_list->expression().size();

      auto allocation =
          allocate_1d_memory(location, n_expr, builder.getI64Type());

      // allocate i64 memref of size n_expr
      // for i in {1,2,3} -> affine.for i : 0 to 3 { element = load(memref, i) }
      int counter = 0;
      for (auto exp : exp_list->expression()) {
        qasm3_expression_generator exp_generator(builder, symbol_table,
                                                 file_name);
        exp_generator.visit(exp);
        auto value = exp_generator.current_value;

        mlir::Value pos =
            get_or_create_constant_integer_value(counter, location);

        builder.create<mlir::StoreOp>(
            location, value, allocation,
            llvm::makeArrayRef(std::vector<mlir::Value>{pos}));

        counter++;
      }

      // Create a new scope for the for loop
      symbol_table.enter_new_scope();

      // Save the current builder point
      auto savept = builder.saveInsertionPoint();

      // Create the for loop
      auto for_loop = builder.create<mlir::AffineForOp>(location, 0, n_expr, 1);

      // Extract the for loop region block and set the insertion point
      mlir::Block& block = *(for_loop.region().getBlocks().begin());
      builder.setInsertionPointToStart(&block);

      // Load the loop variable from the memref allocation
      auto load = builder.create<mlir::LoadOp>(location, allocation,
                                               block.getArgument(0));

      // Save the loaded value as the loop variable name
      symbol_table.add_symbol(idx_var_name, load.result(), {}, true);

      // Visit the for block
      visitChildren(program_block);

      // Exit scope and restore insertion
      symbol_table.exit_scope();
      builder.restoreInsertionPoint(savept);
    } else if (auto range = set_declaration->rangeDefinition()) {
      // this is a range definition
      //     rangeDefinition
      // : LBRACKET expression? COLON expression? ( COLON expression )? RBRACKET
      // ;
      auto is_constant = [](const std::string idx_str) {
        try {
          std::stoi(idx_str);
          return true;
        } catch (std::exception& ex) {
          return false;
        }
      };
 
      // ----------------------------------------//
      // May want to package the following into SymbolTable
      auto all_constants = symbol_table.get_constant_integer_variables();
      std::vector<std::string> variable_names;
      std::vector<double> variable_values;
      for (auto [n, v] : all_constants) {
        variable_names.push_back(n);
        variable_values.push_back(v);
      }
      auto extract_from_variable_idx_str =
          [&](antlr4::ParserRuleContext* idx_node) -> int64_t {
        auto expr_str = idx_node->getText();
        double ref = 0.0;

        symbol_table_t exprtk_symbol_table;
        exprtk_symbol_table.add_constants();
        for (int i = 0; i < variable_names.size(); i++) {
          exprtk_symbol_table.add_variable(variable_names[i],
                                           variable_values[i]);
        }

        expression_t expr;
        expr.register_symbol_table(exprtk_symbol_table);
        parser_t parser;
        if (parser.compile(expr_str, expr)) {
          ref = expr.value();
        } else {
          printErrorMessage("The for range element (" + idx_node->getText() +
                            ") must be a constant integer type.");
        }

        return (int64_t)ref;
      };
      // ----------------------------------------//

      auto range_str =
          range->getText().substr(1, range->getText().length() - 2);
      auto range_elements = split(range_str, ':');
      auto n_expr = range->expression().size();
      int a, b, c;
      if (is_constant(range_elements[0])) {
        a = std::stoi(range_elements[0]);
      } else {
        // infer the constant from the symbol table
        a = extract_from_variable_idx_str(range->expression(0));
      }

      c = 1;
      if (n_expr == 3) {
        if (is_constant(range_elements[2])) {
          b = std::stoi(range_elements[2]);
        } else {
          b = extract_from_variable_idx_str(range->expression(2));
        }
        if (is_constant(range_elements[1])) {
          c = std::stoi(range_elements[1]);
        } else {
          c = extract_from_variable_idx_str(range->expression(1));
        }
      } else {
        if (is_constant(range_elements[1])) {
          b = std::stoi(range_elements[1]);
        } else {
          b = extract_from_variable_idx_str(range->expression(1));
        }
      }
      // std::cout << "A,b,c: " << a << ", " << b << ", " << c << "\n";

      if (a < 0) {
        printErrorMessage("first element of range must be >= 0.");
      }

      if (c < 1) {
        printErrorMessage("step element of range must be > 0.");
      }

      if (b < a) {
        printErrorMessage(
            "end value of range must be greater than start value.");
      }

      // Create a new scope for the for loop
      symbol_table.enter_new_scope();

      // Save the current builder point
      auto savept = builder.saveInsertionPoint();

      // Create the for loop
      auto for_loop = builder.create<mlir::AffineForOp>(location, a, b, c);

      // Extract the for loop region block and set the insertion point
      mlir::Block& block = *(for_loop.region().getBlocks().begin());
      builder.setInsertionPointToStart(&block);

      // Save the loaded value as the loop variable name
      symbol_table.add_symbol(idx_var_name, block.getArgument(0), {}, true);

      // Visit the for block
      visitChildren(program_block);

      // Exit scope and restore insertion
      symbol_table.exit_scope();
      builder.restoreInsertionPoint(savept);

    } else {
      printErrorMessage(
          "For loops must be of form 'for i in {SET}' or 'for i in [RANGE]'.");
    }

  } else {
    // this is a while loop
    auto while_expr = loop_signature->expression();

    // value.dump();
    // printErrorMessage("We do not support while loops at this time.");

    // Create a new scope for the for loop
    symbol_table.enter_new_scope();

    // Save the current builder point
    auto savept = builder.saveInsertionPoint();

    // Mimic a while loop with a for loop over the max number of integer
    // iterations
    auto for_loop = builder.create<mlir::AffineForOp>(
        location, 0, std::numeric_limits<uint64_t>::max(),
        1);  //, llvm::None, for_body_builder);

    // Extract the for loop region block and set the insertion point
    mlir::Block& block = *(for_loop.region().getBlocks().begin());
    builder.setInsertionPointToStart(&block);

    //   // Save the loaded value as the loop variable name
    //   symbol_table.add_symbol(idx_var_name, block.getArgument(0), {}, true);

    // Visit the for block
    visitChildren(program_block);
    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(while_expr);
    auto expr_value = exp_generator.current_value;

    // build up the program block
    auto currRegion = builder.getBlock()->getParent();
    auto savept2 = builder.saveInsertionPoint();
    auto thenBlock = builder.createBlock(currRegion, currRegion->end());
    auto elseBlock = builder.createBlock(currRegion, currRegion->end());

    // Build up the THEN Block, add return at end
    builder.setInsertionPointToStart(thenBlock);
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc(),
                                   llvm::ArrayRef<mlir::Value>());

    // If we have a second program block then we have an else stmt
    builder.setInsertionPointToStart(elseBlock);
    auto yield = builder.create<mlir::AffineYieldOp>(
        location, llvm::makeArrayRef(std::vector<mlir::Value>{}));

    // Restore the insertion point and create the conditional statement
    builder.restoreInsertionPoint(savept2);
    builder.create<mlir::CondBranchOp>(location, expr_value, thenBlock,
                                       elseBlock);

    // Exit scope and restore insertion
    symbol_table.exit_scope();
    builder.restoreInsertionPoint(savept);
  }

  return 0;
}

antlrcpp::Any qasm3_visitor::visitControlDirective(
    qasm3Parser::ControlDirectiveContext* context) {
  auto location = get_location(builder, file_name, context);

  auto stmt = context->getText();

  if (stmt == "break") {
    auto yield = builder.create<mlir::AffineYieldOp>(
        location, llvm::makeArrayRef(std::vector<mlir::Value>{}));
  } else {
    printErrorMessage("we do not yet support the " + stmt +
                      " control directive.");
  }

  return 0;
}
}  // namespace qcor