#pragma once 

#include "Optimizer.hpp"
#include "qcor_utils.hpp"

namespace qcor {

// Re-map xacc Optimizer data types to qcor ones
using OptFunction = xacc::OptFunction;
using Optimizer = xacc::Optimizer;

// Create the desired Optimizer, delegates to xacc getOptimizer
std::shared_ptr<xacc::Optimizer>
createOptimizer(const std::string &type, HeterogeneousMap &&options = {});

}