#pragma once

#include "objective_function.hpp"
#include "qcor_observable.hpp"
#include "qcor_optimizer.hpp"
#include "qcor_utils.hpp"
#include "quantum_kernel.hpp"
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
#ifdef __internal__qcor__compile__plugin__path
  qcor::__internal__::append_plugin_path(__internal__qcor__compile__plugin__path);
#endif
// IMPORTANT: This needs to be set before quantum::initialize
#ifdef __internal__qcor__compile__qrt__mode
    quantum::set_qrt(__internal__qcor__compile__qrt__mode);
#endif
#ifdef __internal__qcor__compile__backend
    quantum::initialize(__internal__qcor__compile__backend, "empty");
#endif
#ifdef __internal__qcor__compile__shots
    quantum::set_shots(std::stoi(__internal__qcor__compile__shots));
#endif
#ifdef __internal__qcor__compile__opt__level
    xacc::internal_compiler::__opt_level =
        __internal__qcor__compile__opt__level;
#endif
#ifdef __internal__qcor__compile__opt__print__stats
    xacc::internal_compiler::__print_opt_stats = true;
#endif
#ifdef __internal__qcor__compile__opt__passes
    xacc::internal_compiler::__user_opt_passes =
        __internal__qcor__compile__opt__passes;
#endif
#ifdef __internal__qcor__compile__placement__name
    xacc::internal_compiler::__placement_name =
        __internal__qcor__compile__placement__name;
#endif
#ifdef __internal__qcor__compile__qubit__map
    xacc::internal_compiler::__qubit_map =
        xacc::internal_compiler::parse_qubit_map(
            __internal__qcor__compile__qubit__map);
#endif
#ifdef __internal__qcor__compile__decorator__list
  xacc::internal_compiler::apply_decorators(__internal__qcor__compile__decorator__list);
#endif

  }
};
internal_startup startup;

} // namespace __internal__

} // namespace qcor
