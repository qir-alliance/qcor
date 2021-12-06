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

#include <string>
#include <vector>

namespace __internal__ {
namespace qcor {
std::string construct_kernel_subtype(
    std::string src_code, const std::string kernel_name,
    const std::vector<std::string> &program_arg_types,
    const std::vector<std::string> &program_parameters,
    std::vector<std::string> bufferNames);
}
}  // namespace __internal__