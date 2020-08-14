#include "taskInitiate.hpp"

namespace qcor {
Handle taskInitiate(std::shared_ptr<ObjectiveFunction> objective,
                          std::shared_ptr<Optimizer> optimizer) {
  return std::async(std::launch::async, [=]() -> ResultsBuffer {
    auto results = optimizer->optimize(*objective.get());
    ResultsBuffer rb;
    rb.q_buffer = objective->get_qreg();
    rb.opt_params = results.second;
    rb.opt_val = results.first;
    return rb;
  });
}
} // namespace qcor