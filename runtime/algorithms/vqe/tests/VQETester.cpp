#include <gtest/gtest.h>

#include "XACC.hpp"
#include "qcor.hpp"
#include "vqe.hpp"
#include "xacc_service.hpp"

using namespace qcor;
using namespace qcor::algorithm;
const std::string rucc = R"rucc(def f(buffer, theta):
    X(0)
    X(1)
    Rx(1.5707,0)
    H(1)
    H(2)
    H(3)
    CNOT(0,1)
    CNOT(1,2)
    CNOT(2,3)
    Rz(theta,3)
    CNOT(2,3)
    CNOT(1,2)
    CNOT(0,1)
    Rx(-1.5707,0)
    H(1)
    H(2)
    H(3)
    )rucc";
TEST(VQETester, checkSimple) {
  if (xacc::hasAccelerator("tnqvm") && xacc::hasCompiler("xacc-py")) {
    auto acc = xacc::getAccelerator("tnqvm");
    auto buffer = acc->createBuffer("q", 4);

    auto compiler = xacc::getService<xacc::Compiler>("xacc-py");

    auto ir = compiler->compile(rucc, nullptr);
    auto ruccsd = ir->getKernel("f");

    auto optimizer = qcor::getOptimizer("nlopt");
    auto observable = qcor::getObservable(
        "(0.174073,0) Z2 Z3 + (0.1202,0) Z1 Z3 + (0.165607,0) Z1 Z2 + "
        "(0.165607,0) Z0 Z3 + (0.1202,0) Z0 Z2 + (-0.0454063,0) Y0 Y1 X2 X3 + "
        "(-0.220041,0) Z3 + (-0.106477,0) + (0.17028,0) Z0 + (-0.220041,0) Z2 "
        "+ (0.17028,0) Z1 + (-0.0454063,0) X0 X1 Y2 Y3 + (0.0454063,0) X0 Y1 "
        "Y2 X3 + (0.168336,0) Z0 Z1 + (0.0454063,0) Y0 X1 X2 Y3");

    VQE vqe;
    vqe.initialize(ruccsd, acc, buffer);
    vqe.execute(*observable.get(), *optimizer.get());
    EXPECT_NEAR(-1.13717, mpark::get<double>(buffer->getInformation("opt-val")), 1e-4);
  }
}

int main(int argc, char **argv) {
  qcor::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
