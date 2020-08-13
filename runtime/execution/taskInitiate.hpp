#pragma once

#include "qcor_utils.hpp"
#include "objective_function.hpp"
#include "qcor_observable.hpp"
#include "qcor_optimizer.hpp"

namespace qcor {

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer);

}