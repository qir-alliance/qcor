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
#include <utility>
#include <vector>
namespace qcor {
namespace utils {
using pauli_decomp_t = std::pair<std::string, double>;
// Input list of quantum gates: name and list of params (e.g. 1 for rx, ry, rz;
// 3 for u3, etc.)
using qop_t = std::pair<std::string, std::vector<double>>;
std::vector<pauli_decomp_t>
decompose_gate_sequence(const std::vector<qop_t> &op_list);
} // namespace utils
} // namespace qcor