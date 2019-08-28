#ifndef RUNTIME_QPU_HANDLER_HPP_
#define RUNTIME_QPU_HANDLER_HPP_

#include "qcor.hpp"

#include "AcceleratorBuffer.hpp"
#include "CompositeInstruction.hpp"
#include "XACC.hpp"
#include <heterogeneous.hpp>

namespace qcor {

class qpu_handler {
protected:
  std::shared_ptr<xacc::AcceleratorBuffer> buffer;

public:
  qpu_handler() = default;
  qpu_handler(std::shared_ptr<xacc::AcceleratorBuffer> b) : buffer(b) {}

  std::shared_ptr<xacc::AcceleratorBuffer> getResults() { return buffer; }

  template <typename QuantumKernel, typename... Args>
  void vqe(QuantumKernel &kernel, std::shared_ptr<Observable> observable,
           std::shared_ptr<Optimizer> optimizer, Args... args) {

    // Turn off backend execution so I can
    // just execute the kernel lambda and get the function
    qcor::switchDefaultKernelExecution(false);

    auto qb = xacc::qalloc(1000);
    auto persisted_function = kernel(qb, args...);

    qcor::switchDefaultKernelExecution(true);

    auto function = xacc::getIRProvider("quantum")->createComposite("f");
    std::istringstream iss(persisted_function);
    function->load(iss);

    auto nLogicalBits = function->nPhysicalBits();
    auto accelerator = xacc::getAccelerator();

    if (!buffer) {
      buffer = xacc::qalloc(nLogicalBits);
    }

    HeterogeneousMap options;
    options.insert("observable", observable);
    options.insert("ansatz", function);
    options.insert("optimizer", optimizer);
    options.insert("accelerator",accelerator);

    auto vqeAlgo = qcor::getAlgorithm("vqe");
    bool success = vqeAlgo->initialize(options);
    if (!success) {
      xacc::error("Error initializing VQE algorithm.");
    }

    vqeAlgo->execute(buffer);
  }

  template <typename QuantumKernel> void execute(QuantumKernel &&kernel) {
    // auto function = qcor::loadCompiledCircuit(kernel());
    // auto nPhysicalQubits = function->nPhysicalBits();

    // auto accelerator = xacc::getAccelerator();

    // if (!buffer) {
    //   buffer = accelerator->createBuffer("q", nPhysicalQubits);
    // }
    // accelerator->execute(buffer, function);
  }

  template <typename QuantumKernel>
  void execute(const std::string &algorithm, QuantumKernel &&kernel) {}
};

} // namespace qcor

#endif