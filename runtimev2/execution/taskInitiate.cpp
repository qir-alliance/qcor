#include "taskInitiate.hpp"

namespace qcor {

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    std::function<double(const std::vector<double>,
                                         std::vector<double> &)> &&opt_function,
                    const int nParameters) {
  return std::async(std::launch::async, [=]() -> ResultsBuffer {
    qcor::OptFunction f(opt_function, nParameters);
    auto results = optimizer->optimize(f);
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &&opt_function) {
  return std::async(std::launch::async, [=, &opt_function]() -> ResultsBuffer {
    auto results = optimizer->optimize(opt_function);
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}

Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                    std::shared_ptr<Optimizer> optimizer,
                    qcor::OptFunction &opt_function) {
  return std::async(std::launch::async, [=, &opt_function]() -> ResultsBuffer {
    auto results = optimizer->optimize(opt_function);
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}
} // namespace qcor