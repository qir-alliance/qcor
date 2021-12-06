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
#include "xacc.hpp"

namespace qcor {
// Descriptions:
// The TracerAccelerator estimates resources (gate counts, qubits) of a quantum
// program by emulating execution the quantum program (not static analysis). The
// idea is to collect/trace a set of metrics during the execution. Measurement
// results can be emulated in different modes: (1) Fixed 0 or 1. (2) Random with
// provided probability threshold. i.e., users can run multiple randomized
// tracing runs to compute the upper bound estimate of resources. This provides
// similar functionalities as the QDK resources estimator
// https://docs.microsoft.com/en-us/azure/quantum/user-guide/machines/resources-estimator
class TracerAccelerator : public xacc::Accelerator {
public:
  // Identifiable interface impls
  virtual const std::string name() const override { return "tracer"; }
  virtual const std::string description() const override { return ""; }

  // Accelerator interface impls
  virtual void initialize(const xacc::HeterogeneousMap &params = {}) override;
  virtual void updateConfiguration(const xacc::HeterogeneousMap &config) override {
    initialize(config);
  };
  virtual const std::vector<std::string> configurationKeys() override {
    return {};
  }
  virtual xacc::HeterogeneousMap getProperties() override { return {}; }
  virtual void execute(std::shared_ptr<xacc::AcceleratorBuffer> buffer,
                       const std::shared_ptr<xacc::CompositeInstruction>
                           compositeInstruction) override;
  virtual void execute(std::shared_ptr<xacc::AcceleratorBuffer> buffer,
                       const std::vector<std::shared_ptr<xacc::CompositeInstruction>>
                           compositeInstructions) override;
  virtual void apply(std::shared_ptr<xacc::AcceleratorBuffer> buffer,
                     std::shared_ptr<xacc::Instruction> inst) override;
  void printResourcesEstimationReport();
private:
  // Probability to measure 1:
  std::unordered_map<size_t, double> qubitIdToMeasureProbs;
  double getMeas1Prob(size_t bitIdx) {
    if (qubitIdToMeasureProbs.find(bitIdx) == qubitIdToMeasureProbs.end()) {
      // Default measurement = 0 => prob gets 1 == 0.0
      return 0.0;
    } else {
      // There is custom measurement probability setting for this qubit.
      return qubitIdToMeasureProbs[bitIdx];
    }
  }
  std::set<size_t> qubit_idxs;
  // Resources metric
  // Gate name to count
  std::unordered_map<std::string, size_t> gateNameToCount;
  bool use_clifford_t = false;
  std::shared_ptr<xacc::CompositeInstruction> counter_composite;
  // TODO:
  // - We could in-principle perform gate layering during tracing
  // i.e., gates on non-overlapping qubits.
  // => add a circuit depth metric
  // - Number of ancilla allocations, etc.
};
} // namespace qcor