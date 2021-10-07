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
#include "taskInitiate.hpp"

namespace qcor {
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                          std::shared_ptr<Optimizer> optimizer) {
  return std::async(std::launch::async, [=]() -> ResultsBuffer {
    auto results = optimizer->optimize(*objective.get());
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}
} // namespace qcor