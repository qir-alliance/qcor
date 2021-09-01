#include "expression_handler.hpp"
#include "exprtk.hpp"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "qasm3_visitor.hpp"

using symbol_table_t = exprtk::symbol_table<double>;
using expression_t = exprtk::expression<double>;
using parser_t = exprtk::parser<double>;

namespace {
/// Creates a single affine "for" loop, iterating from lbs to ubs with
/// the given step.
/// to construct the body of the loop and is passed the induction variable.
mlir::AffineForOp
affineLoopBuilder(mlir::Value lbs_val, mlir::Value ubs_val, int64_t step,
                  std::function<void(mlir::Value)> bodyBuilderFn,
                  mlir::OpBuilder &builder, mlir::Location &loc) {
  if (!ubs_val.getType().isa<mlir::IndexType>()) {
    ubs_val =
        builder.create<mlir::IndexCastOp>(loc, builder.getIndexType(), ubs_val);
  }
  if (!lbs_val.getType().isa<mlir::IndexType>()) {
    lbs_val =
        builder.create<mlir::IndexCastOp>(loc, builder.getIndexType(), lbs_val);
  }
  // Note: Affine for loop only accepts **positive** step:
  // The stride, represented by step, is a positive constant integer which
  // defaults to “1” if not present.
  assert(step != 0);
  if (step > 0) {
    mlir::ValueRange lbs(lbs_val);
    mlir::ValueRange ubs(ubs_val);
    // Create the actual loop
    return builder.create<mlir::AffineForOp>(
        loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
        builder.getMultiDimIdentityMap(ubs.size()), step, llvm::None,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::Value iv, mlir::ValueRange itrArgs) {
          mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv);
          nestedBuilder.create<mlir::AffineYieldOp>(nestedLoc);
        });
  } else {
    // Negative step:
    // a -> b step c (minus)
    // -a -> -b step c (plus) and minus the loop var
    mlir::Value minus_one = builder.create<mlir::ConstantOp>(
        loc, mlir::IntegerAttr::get(lbs_val.getType(), -1));
    lbs_val = builder.create<mlir::MulIOp>(loc, lbs_val, minus_one).result();
    ubs_val = builder.create<mlir::MulIOp>(loc, ubs_val, minus_one).result();
    mlir::ValueRange lbs(lbs_val);
    mlir::ValueRange ubs(ubs_val);
    return builder.create<mlir::AffineForOp>(
        loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
        builder.getMultiDimIdentityMap(ubs.size()), -step, llvm::None,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::Value iv, mlir::ValueRange itrArgs) {
          mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
          mlir::Value minus_one_idx = nestedBuilder.create<mlir::ConstantOp>(
              nestedLoc, mlir::IntegerAttr::get(iv.getType(), -1));
          bodyBuilderFn(
              nestedBuilder.create<mlir::MulIOp>(nestedLoc, iv, minus_one_idx)
                  .result());
          nestedBuilder.create<mlir::AffineYieldOp>(nestedLoc);
        });
  }
}
} // namespace
namespace qcor {
antlrcpp::Any
qasm3_visitor::visitLoopStatement(qasm3Parser::LoopStatementContext *context) {
  auto program_block = context->programBlock();
  const std::string program_block_str = program_block->getText();
  if (program_block_str.find("QCOR_EXPECT_TRUE") != std::string::npos) {
    // QCOR_EXPECT_TRUE will involve early escape (return)
    // hence is not compatible with Region-based dataflow style (Affine/SCF)
    // Since this QCOR_EXPECT_TRUE is for testing only.
    // Don't support it for now.
    // Will need to figure out how to make it work with Affine/SCF.
    printErrorMessage(
        "QCOR_EXPECT_TRUE in loop is not supported now. Stay tuned.", context);
  }
  auto loop_signature = context->loopSignature();
  if (auto membership_test = loop_signature->membershipTest()) {
    // this is a for loop
    auto set_declaration = membership_test->setDeclaration();
    if (set_declaration->LBRACE()) {
      createSetBasedForLoop(context);
    } else if (auto range = set_declaration->rangeDefinition()) {
      // this is a range definition
      //     rangeDefinition
      // : LBRACKET expression? COLON expression? ( COLON expression )? RBRACKET
      // ;
      createRangeBasedForLoop(context);
    } else {
      printErrorMessage(
          "For loops must be of form 'for i in {SET}' or 'for i in [RANGE]'.");
    }
  } else {
    createWhileLoop(context);
  }

  return 0;
}

void qasm3_visitor::createRangeBasedForLoop(
    qasm3Parser::LoopStatementContext *context) {
  auto location = get_location(builder, file_name, context);
  auto loop_signature = context->loopSignature();
  auto program_block = context->programBlock();
  auto membership_test = loop_signature->membershipTest();
  assert(membership_test);
  auto set_declaration = membership_test->setDeclaration();
  assert(set_declaration);
  auto range = set_declaration->rangeDefinition();
  assert(range);
  auto idx_var_name = membership_test->Identifier()->getText();
  // Create Affine loop:
  auto range_str = range->getText().substr(1, range->getText().length() - 2);
  auto range_elements = split(range_str, ':');
  auto n_expr = range->expression().size();
  int a, b, c;

  // First question what type should we use?
  mlir::Type int_type = builder.getI64Type();
  if (symbol_table.has_symbol(range->expression(0)->getText())) {
    int_type =
        symbol_table.get_symbol(range->expression(0)->getText()).getType();
  }
  if (n_expr == 3) {
    if (symbol_table.has_symbol(range->expression(1)->getText())) {
      int_type =
          symbol_table.get_symbol(range->expression(1)->getText()).getType();
    } else if (symbol_table.has_symbol(range->expression(2)->getText())) {
      int_type =
          symbol_table.get_symbol(range->expression(2)->getText()).getType();
    }
  } else {
    if (symbol_table.has_symbol(range->expression(1)->getText())) {
      int_type =
          symbol_table.get_symbol(range->expression(1)->getText()).getType();
    }
  }

  if (int_type.isa<mlir::MemRefType>()) {
    int_type = int_type.cast<mlir::MemRefType>().getElementType();
  }

  c = 1;
  mlir::Value a_value, b_value,
      c_value = get_or_create_constant_integer_value(c, location, int_type,
                                                     symbol_table, builder);

  const auto const_eval_a_val =
      symbol_table.try_evaluate_constant_integer_expression(
          range->expression(0)->getText());
  if (const_eval_a_val.has_value()) {
    // std::cout << "A val = " << const_eval_a_val.value() << "\n";
    a_value = get_or_create_constant_integer_value(
        const_eval_a_val.value(), location, int_type, symbol_table, builder);
  } else {
    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(range->expression(0));
    a_value = exp_generator.current_value;
    if (a_value.getType().isa<mlir::MemRefType>()) {
      a_value = builder.create<mlir::LoadOp>(location, a_value);
    }
  }

  if (n_expr == 3) {
    const auto const_eval_b_val =
        symbol_table.try_evaluate_constant_integer_expression(
            range->expression(2)->getText());
    if (const_eval_b_val.has_value()) {
      // std::cout << "B val = " << const_eval_b_val.value() << "\n";
      b_value = get_or_create_constant_integer_value(
          const_eval_b_val.value(), location, int_type, symbol_table, builder);
    } else {
      qasm3_expression_generator exp_generator(builder, symbol_table,
                                               file_name);
      exp_generator.visit(range->expression(2));
      b_value = exp_generator.current_value;
      if (b_value.getType().isa<mlir::MemRefType>()) {
        b_value = builder.create<mlir::LoadOp>(location, b_value);
      }
    }
    if (symbol_table.has_symbol(range->expression(1)->getText())) {
      printErrorMessage("You must provide loop step as a constant value.",
                        context);
      // c_value = symbol_table.get_symbol(range->expression(1)->getText());
      // c_value = builder.create<mlir::LoadOp>(location, c_value);
      // if (c_value.getType() != int_type) {
      //   printErrorMessage("For loop a, b, and c types are not equal.",
      //                     context, {a_value, c_value});
      // }
    } else {
      c = symbol_table.evaluate_constant_integer_expression(
          range->expression(1)->getText());
      c_value = get_or_create_constant_integer_value(
          c, location, a_value.getType(), symbol_table, builder);
    }

  } else {
    const auto const_eval_b_val =
        symbol_table.try_evaluate_constant_integer_expression(
            range->expression(1)->getText());
    if (const_eval_b_val.has_value()) {
      // std::cout << "B val = " << const_eval_b_val.value() << "\n";
      b_value = get_or_create_constant_integer_value(
          const_eval_b_val.value(), location, int_type, symbol_table, builder);
    } else {
      qasm3_expression_generator exp_generator(builder, symbol_table,
                                               file_name);
      exp_generator.visit(range->expression(1));
      b_value = exp_generator.current_value;
      if (b_value.getType().isa<mlir::MemRefType>()) {
        b_value = builder.create<mlir::LoadOp>(location, b_value);
      }
    }
  }

  // Check if the loop is break-able (contains control directive node)
  const bool isLoopBreakable =
      hasChildNodeOfType<qasm3Parser::ControlDirectiveContext>(*context);
  auto cachedBuilder = builder;
  if (isLoopBreakable) {
    // Add the two loop control bool vars:
    mlir::OpBuilder::InsertionGuard g(builder);
    // Top-level if control (skipping the whole loop if false)
    mlir::Value executeWholeLoop = builder.create<mlir::AllocaOp>(
        location,
        mlir::MemRefType::get(llvm::ArrayRef<int64_t>{}, builder.getI1Type()));
    // Loop body control: skipping portions of the the body if
    // false: e.g., handle 'continue'-like directive.
    mlir::Value executeThisBlock = builder.create<mlir::AllocaOp>(
        location,
        mlir::MemRefType::get(llvm::ArrayRef<int64_t>{}, builder.getI1Type()));
    // store true
    builder.create<mlir::StoreOp>(
        location,
        get_or_create_constant_integer_value(1, location, builder.getI1Type(),
                                             symbol_table, builder),
        executeWholeLoop);
    builder.create<mlir::StoreOp>(
        location,
        get_or_create_constant_integer_value(1, location, builder.getI1Type(),
                                             symbol_table, builder),
        executeThisBlock);
    for_loop_control_vars.push(
        std::make_pair(executeWholeLoop, executeThisBlock));
  }
  // Can use Affine for loop....
  auto forLoop = affineLoopBuilder(
      a_value, b_value, c,
      [&](mlir::Value loop_var) {
        // Create a new scope for the for loop
        symbol_table.enter_new_scope();
        auto loop_var_cast = builder.create<mlir::IndexCastOp>(
            location, builder.getI64Type(), loop_var);
        symbol_table.add_symbol(idx_var_name, loop_var_cast, {}, true);

        if (isLoopBreakable) {
          auto [cond1, cond2] = for_loop_control_vars.top();
          // Wrap/Outline the loop body in an IfOp:
          auto scfIfOp = builder.create<mlir::scf::IfOp>(
              location, mlir::TypeRange(),
              builder.create<mlir::LoadOp>(location, cond1), false);
          auto thenBodyBuilder = scfIfOp.getThenBodyBuilder();
          auto cached_builder = builder;
          builder = thenBodyBuilder;
          visitChildren(program_block);
          builder = cached_builder;
        } else {
          visitChildren(program_block);
        }

        symbol_table.exit_scope();

        if (isLoopBreakable) {
          for_loop_control_vars.pop();
        }
      },
      builder, location);
  builder = cachedBuilder;
}

void qasm3_visitor::createSetBasedForLoop(
    qasm3Parser::LoopStatementContext *context) {
  auto location = get_location(builder, file_name, context);
  auto loop_signature = context->loopSignature();
  auto program_block = context->programBlock();
  auto membership_test = loop_signature->membershipTest();
  assert(membership_test);
  auto set_declaration = membership_test->setDeclaration();
  assert(set_declaration);
  // Must be a set-based for loop
  assert(set_declaration->LBRACE());
  auto idx_var_name = membership_test->Identifier()->getText();
  auto exp_list = set_declaration->expressionList();
  auto n_expr = exp_list->expression().size();

  auto allocation = allocate_1d_memory(location, n_expr, builder.getI64Type());

  // Using Affine For loop:
  // the body builder needs to load the element at the loop index.
  // allocate i64 memref of size n_expr
  // for i in {1,2,3} -> affine.for i : 0 to 3 { element = load(memref, i) }
  int counter = 0;
  for (auto exp : exp_list->expression()) {
    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(exp);
    auto value = exp_generator.current_value;
    mlir::Value pos = get_or_create_constant_index_value(counter, location, 64,
                                                         symbol_table, builder);
    builder.create<mlir::StoreOp>(
        location, value, allocation,
        llvm::makeArrayRef(std::vector<mlir::Value>{pos}));
    counter++;
  }
  auto tmp = get_or_create_constant_index_value(0, location, 64, symbol_table,
                                                builder);
  auto tmp2 = get_or_create_constant_index_value(0, location, 64, symbol_table,
                                                 builder);
  llvm::ArrayRef<mlir::Value> zero_index(tmp2);
  // Loop var must also be an Index type
  // since we'll store the loop index values to this variable.
  auto loop_var_memref = allocate_1d_memory_and_initialize(
      location, 1, builder.getIndexType(), std::vector<mlir::Value>{tmp},
      llvm::makeArrayRef(std::vector<mlir::Value>{tmp}));

  auto a_val = get_or_create_constant_index_value(0, location, 64, symbol_table,
                                                  builder);
  auto b_val = get_or_create_constant_index_value(n_expr, location, 64,
                                                  symbol_table, builder);

  // Check if the loop is break-able (contains control directive node)
  const bool isLoopBreakable =
      hasChildNodeOfType<qasm3Parser::ControlDirectiveContext>(*context);
  auto cachedBuilder = builder;
  if (isLoopBreakable) {
    // Add the two loop control bool vars:
    mlir::OpBuilder::InsertionGuard g(builder);
    // Top-level if control (skipping the whole loop if false)
    mlir::Value executeWholeLoop = builder.create<mlir::AllocaOp>(
        location,
        mlir::MemRefType::get(llvm::ArrayRef<int64_t>{}, builder.getI1Type()));
    // Loop body control: skipping portions of the the body if
    // false: e.g., handle 'continue'-like directive.
    mlir::Value executeThisBlock = builder.create<mlir::AllocaOp>(
        location,
        mlir::MemRefType::get(llvm::ArrayRef<int64_t>{}, builder.getI1Type()));
    // store true
    builder.create<mlir::StoreOp>(
        location,
        get_or_create_constant_integer_value(1, location, builder.getI1Type(),
                                             symbol_table, builder),
        executeWholeLoop);
    builder.create<mlir::StoreOp>(
        location,
        get_or_create_constant_integer_value(1, location, builder.getI1Type(),
                                             symbol_table, builder),
        executeThisBlock);
    for_loop_control_vars.push(
        std::make_pair(executeWholeLoop, executeThisBlock));
  }

  // Can use Affine for loop....
  auto forLoop = affineLoopBuilder(
      a_val, b_val, 1,
      [&](mlir::Value loop_index_var) {
        // Create a new scope for the for loop
        symbol_table.enter_new_scope();
        // Load the value at the index from the set
        auto loop_var =
            builder.create<mlir::LoadOp>(location, allocation, loop_index_var)
                .result();
        // Save the loaded value as the loop variable name
        symbol_table.add_symbol(idx_var_name, loop_var, {}, true);

        if (isLoopBreakable) {
          auto [cond1, cond2] = for_loop_control_vars.top();
          // Wrap/Outline the loop body in an IfOp:
          auto scfIfOp = builder.create<mlir::scf::IfOp>(
              location, mlir::TypeRange(),
              builder.create<mlir::LoadOp>(location, cond1), false);
          auto thenBodyBuilder = scfIfOp.getThenBodyBuilder();
          auto cached_builder = builder;
          builder = thenBodyBuilder;
          visitChildren(program_block);
          builder = cached_builder;
        } else {
          visitChildren(program_block);
        }
        symbol_table.exit_scope();

        if (isLoopBreakable) {
          for_loop_control_vars.pop();
        }
      },
      builder, location);
  builder = cachedBuilder;
}

void qasm3_visitor::createWhileLoop(
    qasm3Parser::LoopStatementContext *context) {
  auto location = get_location(builder, file_name, context);
  auto loop_signature = context->loopSignature();
  auto program_block = context->programBlock();
  assert(loop_signature->booleanExpression());

  // FIXME: convert to mlir::scf::WhileOp (should be easy since it's
  // conditioned on an i1 value)
  // this is a while loop
  auto while_expr = loop_signature->booleanExpression();

  // Create a new scope for the for loop
  symbol_table.enter_new_scope();

  auto currRegion = builder.getBlock()->getParent();

  auto savept = builder.saveInsertionPoint();
  auto headerBlock = builder.createBlock(currRegion, currRegion->end());
  auto bodyBlock = builder.createBlock(currRegion, currRegion->end());
  auto exitBlock = builder.createBlock(currRegion, currRegion->end());

  builder.restoreInsertionPoint(savept);
  builder.create<mlir::BranchOp>(location, headerBlock);

  builder.setInsertionPointToEnd(headerBlock);
  qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
  exp_generator.visit(while_expr);
  auto expr_value = exp_generator.current_value;
  builder.create<mlir::CondBranchOp>(location, expr_value, bodyBlock,
                                     exitBlock);

  builder.setInsertionPointToStart(bodyBlock);
  current_loop_exit_block = exitBlock;
  current_loop_header_block = headerBlock;

  visitChildren(program_block);

  current_loop_header_block = nullptr;
  current_loop_exit_block = nullptr;

  builder.create<mlir::BranchOp>(location, headerBlock);
  builder.setInsertionPointToStart(exitBlock);

  symbol_table.exit_scope();

  // This is where we do some manipulation of
  // the basic blocks, lets store the current last block
  // so that finalize_mlirgen() can add return and deallocs
  // correctly

  symbol_table.set_last_created_block(exitBlock);
}

} // namespace qcor