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


// // Controlled-Op transform:
// // Usage: Controlled::Apply(controlBit, QuantumKernel, Args...)
// // where Args... are arguments that will be passed to the kernel.
// // Note: we use a class with a static member function to enforce
// // that the invocation is Controlled::Apply(...) (with the '::' separator),
// // hence the XASM compiler cannot mistakenly parse this as a XASM call.
// class Controlled {
// public:
//   template <typename FunctorType, typename... ArgumentTypes>
//   static void Apply(int ctrlIdx, FunctorType functor, ArgumentTypes... args) {
//     __controlledIdx = {ctrlIdx};
//     const auto __cached_execute_flag = __execute;
//     __execute = false;
//     functor(args...);
//     __controlledIdx.clear();
//     __execute = __cached_execute_flag;
//   }
// };

// template <typename QuantumKernel, typename... Args>
// std::function<void(Args...)> measure_all(QuantumKernel &kernel, Args... args) {
//   return [&](Args... args) {
//     auto internal_kernel =
//         qcor::__internal__::kernel_as_composite_instruction(kernel, args...);
//     auto q = std::get<0>(std::forward_as_tuple(args...));
//     auto q_name = q.name();
//     auto nq = q.size();
//     auto observable = allZs(nq);
//     auto observed = observable.observe(internal_kernel)[0];
//     auto visitor = std::make_shared<xacc_to_qrt_mapper>(q_name);
//     quantum::clearProgram();
//     xacc::InstructionIterator iter(observed);
//     while (iter.hasNext()) {
//       auto next = iter.next();
//       if (!next->isComposite()) {
//         next->accept(visitor);
//       }
//     }
//     if (xacc::internal_compiler::__execute) {
//       ::quantum::submit(q.results());
//     }
//     return;
//   };
// }

// template <typename QuantumKernel, typename... Args>
// std::function<void(Args...)>
// apply_transformations(QuantumKernel &kernel,
//                       std::vector<std::string> &&transforms, Args... args) {
//   auto internal_kernel =
//       qcor::__internal__::kernel_as_composite_instruction(kernel, args...);

//   for (auto &transform : transforms) {

//     auto xacc_transform = qcor::__internal__::get_transformation(transform);
//     xacc_transform->apply(internal_kernel, xacc::internal_compiler::get_qpu());
//   }

//   return [internal_kernel](Args... args) {
//     // map back to executable kernel
//     quantum::clearProgram();
//     auto q = std::get<0>(std::forward_as_tuple(args...));
//     auto q_name = q.name();
//     auto visitor = std::make_shared<xacc_to_qrt_mapper>(q_name);
//     xacc::InstructionIterator iter(internal_kernel);
//     while (iter.hasNext()) {
//       auto next = iter.next();
//       if (!next->isComposite()) {
//         next->accept(visitor);
//       }
//     }
//     if (xacc::internal_compiler::__execute) {
//       ::quantum::submit(q.results());
//     }
//   };
// }

// template <typename QuantumKernel, typename... Args>
// const std::size_t n_instructions(QuantumKernel &kernel, Args... args) {
//   return qcor::__internal__::kernel_as_composite_instruction(kernel, args...)
//       ->nInstructions();
// }

// template <typename QuantumKernel, typename... Args>
// const std::size_t depth(QuantumKernel &kernel, Args... args) {
//   return qcor::__internal__::kernel_as_composite_instruction(kernel, args...)
//       ->depth();
// }

} // namespace qcor

