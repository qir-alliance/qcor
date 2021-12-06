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
#include "qcor_qsim.hpp"

namespace qcor {
namespace QuaSiMo {
std::shared_ptr<CostFunctionEvaluator>
getEvaluator(Operator *observable, const HeterogeneousMap &params);

// Implements Prony method for IQPE signal fitting.
// i.e. fit g(t) = sum a_i * exp(i * omega_i * t)
// Returns the vector of amplitude {a_i} and freq. {omega_i}
// Returns a vector of <Ampl, Freq> pair
using PronyResult =
    std::vector<std::pair<std::complex<double>, std::complex<double>>>;
PronyResult pronyFit(const std::vector<std::complex<double>> &in_signal);
} // namespace QuaSiMo
} // namespace qcor