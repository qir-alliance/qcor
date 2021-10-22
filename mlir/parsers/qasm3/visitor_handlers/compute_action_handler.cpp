/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitCompute_action_stmt(
    qasm3Parser::Compute_action_stmtContext *context) {
  auto location = get_location(builder, file_name, context);

  builder.create<mlir::quantum::ComputeMarkerOp>(location);
  visit(context->compute_block);
  builder.create<mlir::quantum::ComputeUnMarkerOp>(location);
  visit(context->action_block);

  builder.create<mlir::quantum::ComputeMarkerOp>(location);

  const auto createFuncCall = [](const std::string &func_name,
                                 mlir::OpBuilder &opBuilder,
                                 mlir::ModuleOp &parentModule) {
    mlir::FlatSymbolRefAttr funcRef = [&]() {
      mlir::OpBuilder::InsertionGuard guard(opBuilder);
      opBuilder.setInsertionPointToStart(
          &parentModule.getRegion().getBlocks().front());
      if (parentModule.lookupSymbol<mlir::FuncOp>(func_name)) {
        auto fnNameAttr = opBuilder.getSymbolRefAttr(func_name);
        return fnNameAttr;
      }

      auto func_decl = opBuilder.create<mlir::FuncOp>(
          opBuilder.getUnknownLoc(), func_name,
          opBuilder.getFunctionType(llvm::None, llvm::None));
      func_decl.setVisibility(mlir::SymbolTable::Visibility::Private);
      return mlir::SymbolRefAttr::get(func_name, parentModule->getContext());
    }();

    opBuilder.create<mlir::CallOp>(opBuilder.getUnknownLoc(), funcRef,
                                   llvm::None, llvm::None);
  };

  // Direct injection of Adj start/end functions.
  // !!FIXME!!: This compute_block is very generic, it'd be hard to infer all the qubit operands.
  // hence, we cannot enclose it in a modifier-scoped block yet.
  // SSA tracking (for optimization) for compute/action will not be guaranteed.
  createFuncCall("__quantum__rt__start_adj_u_region", builder, m_module);
  visit(context->compute_block);
  createFuncCall("__quantum__rt__end_adj_u_region", builder, m_module);

  builder.create<mlir::quantum::ComputeUnMarkerOp>(location);

  return 0;
}

}  // namespace qcor