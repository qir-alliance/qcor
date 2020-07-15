#pragma once

#include "qcor_utils.hpp"
#include "objective_function.hpp"
#include "qcor_observable.hpp"
#include "qcor_optimizer.hpp"

namespace qcor {

// Execute asynchronous task - find optimal value of the given objective function
// using the provided optimizer, and custom std::function OptFunction delegating 
// to the ObjectiveFunction. Provide the number of variational parameters
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    std::function<double(const std::vector<double>,
                                         std::vector<double> &)> &&opt_function,
                    const int nParameters);

// Execute asynchronous task - find optimal value of the given objective function
// using the provided optimizer OptFunction delegating 
// to the ObjectiveFunction.
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &&opt_function);

// Execute asynchronous task - find optimal value of the given objective function
// using the provided optimizer OptFunction delegating 
// to the ObjectiveFunction.
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &opt_function);

// Execute asynchronous task - find optimal value of the given objective function
// using the provided optimizer, and custom TranslationFunctor, which will map 
// vector<double> to the required ObjectiveFunction arguments. Provide the number
// variational parameters
template <typename... Args>
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    TranslationFunctor<Args...> translation,
                    const int nParameters) {
  return taskInitiate(
      objective, optimizer,
      [=](const std::vector<double> x, std::vector<double> &dx) {
        auto translated_tuple = translation(x);
        return qcor::__internal__::call(objective, translated_tuple);
      },
      nParameters);
}

// Execute asynchronous task - find optimal value of the given objective function
// using the provided optimizer, and custom TranslationFunctor, which will map 
// vector<double> to the required ObjectiveFunction arguments. Provide the number
// variational parameters and GradientEvaluator
template <typename... Args>
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    GradientEvaluator &grad_evaluator,
                    TranslationFunctor<Args...> translation,
                    const int nParameters) {
  return taskInitiate(
      objective, optimizer,
      [=](const std::vector<double> x, std::vector<double> &dx) {
        grad_evaluator(x, dx);
        auto translated_tuple = translation(x);
        return qcor::__internal__::call(objective, translated_tuple);
      },
      nParameters);
}
}