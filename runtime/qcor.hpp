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

#include "qcor_lang_ext.hpp"

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
#ifdef __internal__qcor__print__final__submission
    xacc::internal_compiler::__print_final_submission = true;
#endif
#ifdef __internal__qcor__validate__execution
    xacc::internal_compiler::__validate_nisq_execution = true;
#endif
#ifdef __internal__qcor__autograd__method
    xacc::internal_compiler::set_autograd_method(__internal__qcor__autograd__method);
#endif
  }
};
internal_startup startup;

} // namespace __internal__

} // namespace qcor

#define qcor_include_qasm(NAME) extern "C" void NAME(qreg);
#define qcor_include_qsharp(NAME, RETURN_TYPE, ...) extern "C" RETURN_TYPE NAME(...);
