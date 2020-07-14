#pragma once

#include "qcor_utils.hpp"
#include "objective_function.hpp"
#include "qcor_observable.hpp"
#include "qcor_optimizer.hpp"

namespace qcor {

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    std::function<double(const std::vector<double>,
                                         std::vector<double> &)> &&opt_function,
                    const int nParameters);

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &&opt_function);

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &opt_function);

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