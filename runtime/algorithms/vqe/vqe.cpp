#include "vqe.hpp"

#include "Observable.hpp"
#include "XACC.hpp"
#include "optimizer.hpp"

#include "InstructionParameter.hpp"
#include <memory>

#include "PauliOperator.hpp"
using namespace xacc;

namespace qcor {
namespace algorithm {
void VQE::execute(xacc::Observable &observable, Optimizer &optimizer) {

  // Here we just need to make a lambda kernel
  // to optimize that makes calls to the targeted QPU.
  OptFunction f(
      [&](const std::vector<double> &x) {
        auto kernels = observable.observe(kernel);
        std::vector<double> coefficients;
        std::vector<std::string> kernelNames;
        std::vector<std::shared_ptr<Function>> fsToExec;

        double identityCoeff = 0.0;
        for (auto &f : kernels) {
          kernelNames.push_back(f->name());
          InstructionParameter p = f->getOption("coefficient");
          std::complex<double> coeff = p.as<std::complex<double>>();

          if (f->nInstructions() > kernel->nInstructions()) {
            fsToExec.push_back(f->operator()(x));
            coefficients.push_back(std::real(coeff));
          } else {
            identityCoeff += std::real(coeff);
          }
        }

        auto buffers = accelerator->execute(buffer, fsToExec);

        double energy = identityCoeff;
        for (int i = 0; i < buffers.size(); i++) {
          auto expval = buffers[i]->getExpectationValueZ();
          energy += expval * coefficients[i];
          buffers[i]->addExtraInfo("coefficient", coefficients[i]);
          buffers[i]->addExtraInfo("kernel", fsToExec[i]->name());
          buffers[i]->addExtraInfo("exp-val-z",
                                   expval);
          buffers[i]->addExtraInfo("parameters", x);
          buffer->appendChild(fsToExec[i]->name(), buffers[i]);
        }

        std::stringstream ss;
        ss << "E(" << x[0];
        for (int i = 1; i < x.size(); i++)
          ss << "," << x[i];
        ss << ") = " << energy;
        xacc::info(ss.str());
        return energy;
      },
      kernel->nParameters());

  // std::cout << f({.55}) << "\n";
  auto result = optimizer.optimize(f);

  buffer->addExtraInfo("opt-val", ExtraInfo(result.first));
  buffer->addExtraInfo("opt-params", ExtraInfo(result.second));
  return;
}
} // namespace algorithm
} // namespace qcor