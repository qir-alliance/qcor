#pragma once
#include <thread>             
#include <mutex>              
#include <condition_variable> 

namespace qcor {
// Experimental optimization stepper to guide the Q# VQE loop.
class OptStepper {
public:
  OptStepper(const std::string &in_optName,
             xacc::HeterogeneousMap in_config = {}) {
    m_optimizer = qcor::createOptimizer(in_optName, std::move(in_config));
  }
  // Call by the main thread to update the  params and cost val...
  void update(const std::vector<double> &in_params, double in_costVal) {
    m_nextParamsAvail = false;
    if (m_dim == 0) {
      // First update:
      // Assuming the initial params are set by the caller,
      // i.e. this stepper doesn't control initial values.
      m_dim = in_params.size();
      m_optimizer->appendOption("initial-parameters", in_params);
      m_optFn = xacc::OptFunction(
          [&](const std::vector<double> &x, std::vector<double> &g) {
            return this->eval(x);
          },
          m_dim);
      m_currentCostVal = in_costVal;
      m_optParams = in_params;
      m_resultAvail = true;
      m_optThread = std::thread([&]() {
        auto result = m_optimizer->optimize(m_optFn);
        m_optDone = true;
      });
      return;
    }

    // A new iteration...
    m_optParams = in_params;
    m_currentCostVal = in_costVal;
    m_resultAvail = true;
    // Notify that the new result is avail...
    m_cv.notify_all();
  }

  // Fake evaluator function for the optimizer:
  // Run on the optimizer thread...
  double eval(const std::vector<double> &x) {
    if (!m_resultAvail) {
      m_optParams = x;
    }

    // Wait for the result to be available,
    // update() was called by the main thread.
    std::unique_lock<std::mutex> lck(m_mtx);
    m_cv.wait(lck, [this] { return m_resultAvail; });
    // This result has been processed...
    m_resultAvail = false;
    m_nextParamsAvail = true;
    return m_currentCostVal;
  }

  // Call by main thread...
  std::vector<double> getNextParams() {
    if (m_optDone) {
      // Return empty to signal that the optimizer has done.
      m_optThread.join();
      return {};
    }
    // Wait for the optimizer to give us a new set of parameters.
    // This should be fast, hence just poll.
    while (!m_nextParamsAvail) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return m_optParams;
  }

private:
  int m_dim = 0;
  std::shared_ptr<xacc::Optimizer> m_optimizer;
  std::vector<double> m_optParams;
  double m_currentCostVal;
  bool m_resultAvail = false;
  bool m_nextParamsAvail = false;
  bool m_optDone = false;
  // Optimizer thread
  std::mutex m_mtx;
  std::condition_variable m_cv;
  std::thread m_optThread;
  xacc::OptFunction m_optFn;
};
} // namespace qcor
