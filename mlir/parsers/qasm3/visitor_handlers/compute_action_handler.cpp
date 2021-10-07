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
  builder.create<mlir::quantum::StartAdjointURegion>(location);
  visit(context->compute_block);
  builder.create<mlir::quantum::EndAdjointURegion>(location);
  builder.create<mlir::quantum::ComputeUnMarkerOp>(location);

  return 0;
}

}  // namespace qcor