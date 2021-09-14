#include "xacc.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include "mirror_circuit_rb.hpp"
#include "qcor_ir.hpp"
#include <random>
namespace {
double random_angle() {
  static std::uniform_real_distribution<double> dis(-M_PI, M_PI);
  static std::default_random_engine re;
  return dis(re);
}
} // namespace

TEST(MirrorCircuitTester, checkU3Inverse) {
  auto provider = xacc::getIRProvider("quantum");
  constexpr int NUM_TESTS = 1000;
  for (int i = 0; i < NUM_TESTS; ++i) {
    auto circuit = provider->createComposite("test");
    const double theta = random_angle();
    const double phi = random_angle();
    const double lambda = random_angle();
    circuit->addInstruction(provider->createInstruction(
        "U", {0}, std::vector<xacc::InstructionParameter>{theta, phi, lambda}));
    auto [mirror_cir, expected_result] = qcor::createMirrorCircuit(
        std::make_shared<qcor::CompositeInstruction>(circuit));
    auto accelerator = xacc::getAccelerator("qpp", {{"shots", 1024}});
    // circuit->addInstructions(mirror_cir->getInstructions());
    circuit->addInstruction(provider->createInstruction("Measure", {0}));
    // std::cout << "HOWDY: \n" << circuit->toString() << "\n";
    auto buffer = xacc::qalloc(1);
    accelerator->execute(buffer, circuit);
    buffer->print();
    EXPECT_EQ(buffer->getMeasurementCounts().size(), 1);
    EXPECT_EQ(buffer->getMeasurementCounts()["0"], 1024);
  }
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
