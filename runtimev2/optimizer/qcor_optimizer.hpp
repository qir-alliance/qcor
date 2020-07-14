#pragma once 

#include "Optimizer.hpp"
#include "qcor_utils.hpp"

namespace qcor {

using OptFunction = xacc::OptFunction;
using Optimizer = xacc::Optimizer;

// Create the desired Optimizer
std::shared_ptr<xacc::Optimizer>
createOptimizer(const std::string &type, HeterogeneousMap &&options = {});

}