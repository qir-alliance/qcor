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
  vqe->initialize(observable, ruccsd.get());
  vqe->set_qreg(buffer);

  qcor::OptFunction f(
      [&](const std::vector<double> x, std::vector<double> &grad) {
        return (*vqe)(buffer, x[0]);
      },
      1);
  auto results = optimizer->optimize(f);

  EXPECT_NEAR(-1.13717, results.first, 1e-4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
