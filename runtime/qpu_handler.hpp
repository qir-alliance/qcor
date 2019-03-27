#ifndef RUNTIME_QPU_HANDLER_HPP_
#define RUNTIME_QPU_HANDLER_HPP_

#include "qcor.hpp"

#include "AcceleratorBuffer.hpp"
#include "Function.hpp"
#include "InstructionIterator.hpp"
#include "InstructionParameter.hpp"
#include "Observable.hpp"
#include "XACC.hpp"

#include <complex>
#include <string>

namespace qcor {

class qpu_handler {

protected:
  std::shared_ptr<xacc::AcceleratorBuffer> buffer;

public:
  std::shared_ptr<xacc::AcceleratorBuffer> getResults() { return buffer; }

  template <typename QuantumKernel>
  void vqe(QuantumKernel &&kernel, xacc::Observable &observable,
           std::shared_ptr<Optimizer> optimizer) {

    auto function = qcor::loadCompiledCircuit(kernel());
    auto nPhysicalQubits = function->nPhysicalBits();

    auto accelerator = xacc::getAccelerator();
    buffer = accelerator->createBuffer("q", nPhysicalQubits);

    // Here we just need to make a lambda function
    // to optimize that makes calls to the targeted QPU.
    OptFunction f(
        [&](const std::vector<double> &x) {
          auto functions = observable.observe(function);
          std::vector<double> coefficients;
          std::vector<std::string> functionNames;
          std::vector<std::shared_ptr<Function>> fsToExec;

          double identityCoeff = 0.0;
          for (auto &f : functions) {
            functionNames.push_back(f->name());
            InstructionParameter p = f->getOption("coefficient");
            std::complex<double> coeff = p.as<std::complex<double>>();

            if (f->nInstructions() > function->nInstructions()) {
              fsToExec.push_back(f->operator()(x));
              coefficients.push_back(std::real(coeff));
            } else {
              identityCoeff += std::real(coeff);
            }
          }

          auto buffers = accelerator->execute(buffer, fsToExec);

          double energy = identityCoeff;
          for (int i = 0; i < buffers.size(); i++) {
            energy += buffers[i]->getExpectationValueZ() * coefficients[i];
            buffers[i]->addExtraInfo("coefficient", coefficients[i]);
            buffers[i]->addExtraInfo("kernel", fsToExec[i]->name());
            buffers[i]->addExtraInfo("exp-val-z", buffers[i]->getExpectationValueZ());
            buffers[i]->addExtraInfo("parameters", x);
            buffer->appendChild(fsToExec[i]->name(), buffers[i]);
          }
          xacc::info("E("+std::to_string(x[0])+") = " + std::to_string(energy));
          return energy;
        },
        function->nParameters());

    // std::cout << f({.55}) << "\n";
    auto result = optimizer->optimize(f);

    buffer->addExtraInfo("opt-val", ExtraInfo(result.first));
    buffer->addExtraInfo("opt-params", ExtraInfo(result.second));

    return;
  }

  template <typename QuantumKernel> void execute(QuantumKernel &&kernel) {
    auto function = qcor::loadCompiledCircuit(kernel());
    auto nPhysicalQubits = function->nPhysicalBits();

    // auto function = kernel();
    auto accelerator = xacc::getAccelerator();
    buffer = accelerator->createBuffer("q", nPhysicalQubits);
    accelerator->execute(buffer, function);
  }

  template <typename QuantumKernel>
  void execute(const std::string &algorithm, QuantumKernel &&kernel) {}
};

} // namespace qcor

#endif