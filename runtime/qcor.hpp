#pragma once

#include "qcor_utils.hpp"
#include "qcor_observable.hpp"
#include "qcor_optimizer.hpp"
#include "quantum_kernel.hpp"
#include "objective_function.hpp"
#include "taskInitiate.hpp"
#include <qalloc>

namespace qcor {

namespace __internal__ {
// This class gives us a way to
// run some startup routine before
// main(). Specifically we use it to ensure that
// the accelerator backend is set in the event no
// quantum kernels are found by the syntax handler.
class internal_startup {
public:
  internal_startup() {
#ifdef __internal__qcor__compile__backend
    quantum::initialize(__internal__qcor__compile__backend, "empty");
#endif
  }
};
internal_startup startup;

} // namespace __internal__

} // namespace qcor

