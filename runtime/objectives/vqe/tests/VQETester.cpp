#include "qcor.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_quantum_gate_api.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>

using namespace xacc;
const std::string rucc = R"rucc(__qpu__ void f(qbit q, double t0) {
    X(q[0]);
    X(q[1]);
    Rx(q[0],1.5707);
    H(q[1]);
    H(q[2]);
    H(q[3]);
    CNOT(q[0],q[1]);
    CNOT(q[1],q[2]);
    CNOT(q[2],q[3]);
    Rz(q[3], t0);
    CNOT(q[2],q[3]);
    CNOT(q[1],q[2]);
    CNOT(q[0],q[1]);
    Rx(q[0],-1.5707);
    H(q[1]);
    H(q[2]);
    H(q[3]);
})rucc";

const std::string qaoa_ansatz_src = R"##(
  __qpu__ void qaoa_ansatz(qreg q, int n, std::vector<double> betas, std::vector<double> gammas, std::shared_ptr<xacc::Observable> costHamiltonian, std::shared_ptr<xacc::Observable> refHamiltonian) {
    // Call the qaoa circuit
    qaoa(q,n,betas,gammas,costHamiltonian,refHamiltonian); 
  })##";

TEST(VQETester, checkSimple) {

  xacc::internal_compiler::compiler_InitializeXACC("qpp");

  auto buffer = qalloc(4);

  auto ruccsd = xacc::getService<xacc::Compiler>("xasm")
                    ->compile(rucc, nullptr)
                    ->getComposite("f");

  auto optimizer = qcor::createOptimizer("nlopt");
  std::shared_ptr<Observable> observable = xacc::quantum::getObservable(
      "pauli",
      std::string(
          "(0.174073,0) Z2 Z3 + (0.1202,0) Z1 Z3 + (0.165607,0) Z1 Z2 + "
          "(0.165607,0) Z0 Z3 + (0.1202,0) Z0 Z2 + (-0.0454063,0) Y0 Y1 X2 X3 "
          "+ "
          "(-0.220041,0) Z3 + (-0.106477,0) + (0.17028,0) Z0 + (-0.220041,0) "
          "Z2 "
          "+ (0.17028,0) Z1 + (-0.0454063,0) X0 X1 Y2 Y3 + (0.0454063,0) X0 Y1 "
          "Y2 X3 + (0.168336,0) Z0 Z1 + (0.0454063,0) Y0 X1 X2 Y3"));

  auto vqe = xacc::getService<qcor::ObjectiveFunction>("vqe");
  vqe->initialize(observable.get(), ruccsd);
  vqe->set_qreg(buffer);

  qcor::OptFunction f(
      [&](const std::vector<double> x, std::vector<double> &grad) {
        return (*vqe)(buffer, x[0]);
      },
      1);
  auto results = optimizer->optimize(f);

  EXPECT_NEAR(-1.13717, results.first, 1e-4);
}

TEST(VQETester, checkQaoa) {
  xacc::internal_compiler::compiler_InitializeXACC("qpp");
  auto buffer = qalloc(2);
  auto qaoaCirc = xacc::getService<xacc::Compiler>("xasm")
                  ->compile(qaoa_ansatz_src, nullptr)
                  ->getComposite("qaoa_ansatz");
                  
  auto optimizer = qcor::createOptimizer("nlopt");
  std::shared_ptr<Observable> observable = xacc::quantum::getObservable(
      "pauli",
      std::string(
          "5.907 - 2.1433 X0X1 "
          "- 2.1433 Y0Y1"
          "+ .21829 Z0 - 6.125 Z1"));

  std::shared_ptr<Observable> refHamiltonian = xacc::quantum::getObservable(
      "pauli",
      std::string("1.0 X0 + 1.0 X1")
  );

  auto vqe = xacc::getService<qcor::ObjectiveFunction>("vqe");
  vqe->initialize(observable, qaoaCirc);
  vqe->set_qreg(buffer);

  const int nbSteps = 2;
  const int nbParamsPerStep = 2 /*beta (mixer)*/ + 4 /*gamma (cost)*/;
  const int totalParams = nbSteps * nbParamsPerStep;
  int iterCount = 0;

  qcor::OptFunction f(
      [&](const std::vector<double> x, std::vector<double> &grad) {
        std::vector<double> betas;
        std::vector<double> gammas;
        // Unpack nlopt params
        // Beta: nbSteps * number qubits
        for (int i = 0; i < nbSteps * buffer.size(); ++i) {
          betas.emplace_back(x[i]);
        }

        for (int i = betas.size(); i < x.size(); ++i) {
          gammas.emplace_back(x[i]);
        }

        const double costVal = (*vqe)(buffer, buffer.size(), betas, gammas, observable.get(), refHamiltonian.get());
        std::cout << "Iter " << iterCount << ": Cost = " << costVal << "\n";
        iterCount++;
        return costVal;
      },
      totalParams);
  auto results = optimizer->optimize(f);
  std::cout << "Final cost: " << results.first << "\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
