#include "xacc.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include "mirror_circuit_rb.hpp"
#include "qcor_ir.hpp"
#include <random>
#include "NoiseModel.hpp"

namespace {
double random_angle() {
  static std::uniform_real_distribution<double> dis(-M_PI, M_PI);
  static std::default_random_engine re;
  return dis(re);
}
} // namespace

TEST(MirrorCircuitTester, checkU3Inverse) {
  auto provider = xacc::getIRProvider("quantum");
  constexpr int NUM_TESTS = 100;
  auto accelerator = xacc::getAccelerator("qpp", {{"shots", 1024}});
  std::set<std::string> allBitStrings;
  for (int i = 0; i < NUM_TESTS; ++i) {
    auto circuit = provider->createComposite(std::string("test") + std::to_string(i));
    const double theta = random_angle();
    const double phi = random_angle();
    const double lambda = random_angle();
    circuit->addInstruction(provider->createInstruction(
        "U", {0}, std::vector<xacc::InstructionParameter>{theta, phi, lambda}));
    auto [mirror_cir, expected_result] = qcor::MirrorCircuitValidator::createMirrorCircuit(
        std::make_shared<qcor::CompositeInstruction>(circuit));
    EXPECT_EQ(expected_result.size(), 1);
    EXPECT_EQ(mirror_cir->nInstructions(), 2);
    // std::cout << "Expected: " << expected_result[0] << "\n";
    // std::cout << "HOWDY: \n" << mirror_cir->toString() << "\n";
    auto mirror_circuit = provider->createComposite("test_mirror");
    mirror_circuit->addInstructions(mirror_cir->getInstructions());
    mirror_circuit->addInstruction(provider->createInstruction("Measure", {0}));
    auto mc_buffer = xacc::qalloc(1);
    accelerator->execute(mc_buffer, mirror_circuit);
    // mc_buffer->print();
    EXPECT_EQ(mc_buffer->getMeasurementCounts().size(), 1);
    EXPECT_EQ(
        mc_buffer->getMeasurementCounts()[std::to_string(expected_result[0])],
        1024);
    allBitStrings.emplace(std::to_string(expected_result[0]));
  }
  // Cover both cases (randomized Pauli worked)
  EXPECT_EQ(allBitStrings.size(), 2);
}

// Layer of U3's on multiple qubits
TEST(MirrorCircuitTester, checkMultipleU3) {
  auto provider = xacc::getIRProvider("quantum");
  constexpr int NUM_TESTS = 100;
  auto accelerator = xacc::getAccelerator("qpp", {{"shots", 1024}});
  std::set<std::string> allBitStrings;
  for (int i = 0; i < NUM_TESTS; ++i) {
    auto circuit = provider->createComposite(std::string("test") + std::to_string(i));
    const double theta1 = random_angle();
    const double phi1 = random_angle();
    const double lambda1 = random_angle();
    circuit->addInstruction(provider->createInstruction(
        "U", {0}, std::vector<xacc::InstructionParameter>{theta1, phi1, lambda1}));

    const double theta2 = random_angle();
    const double phi2 = random_angle();
    const double lambda2 = random_angle();
    circuit->addInstruction(provider->createInstruction(
        "U", {1}, std::vector<xacc::InstructionParameter>{theta2, phi2, lambda2}));
    auto [mirror_cir, expected_result] = qcor::MirrorCircuitValidator::createMirrorCircuit(
        std::make_shared<qcor::CompositeInstruction>(circuit));
    EXPECT_EQ(expected_result.size(), 2);
    EXPECT_EQ(mirror_cir->nInstructions(), 4);
    // std::cout << "Expected: " << expected_result[0] << expected_result[1] << "\n";
    // std::cout << "HOWDY: \n" << mirror_cir->toString() << "\n";
    auto mirror_circuit = provider->createComposite("test_mirror");
    mirror_circuit->addInstructions(mirror_cir->getInstructions());
    mirror_circuit->addInstruction(provider->createInstruction("Measure", {0}));
    mirror_circuit->addInstruction(provider->createInstruction("Measure", {1}));
    auto mc_buffer = xacc::qalloc(2);
    accelerator->execute(mc_buffer, mirror_circuit);
    // mc_buffer->print();
    const std::string expectedBitString =
        std::to_string(expected_result[0]) + std::to_string(expected_result[1]);
    EXPECT_EQ(mc_buffer->getMeasurementCounts().size(), 1);
    EXPECT_EQ(mc_buffer->getMeasurementCounts()[expectedBitString], 1024);
    allBitStrings.emplace(expectedBitString);
  }
  // We should have seen all 4 possible cases with that number of randomized
  // Pauli runs
  EXPECT_EQ(allBitStrings.size(), 4);
}

TEST(MirrorCircuitTester, checkCliffordGates) {
  auto provider = xacc::getIRProvider("quantum");
  constexpr int NUM_TESTS = 100;
  auto accelerator = xacc::getAccelerator("qpp", {{"shots", 1024}});
  std::set<std::string> allBitStrings;
  for (int i = 0; i < NUM_TESTS; ++i) {
    auto circuit = provider->createComposite(std::string("test") + std::to_string(i));
    const double theta1 = random_angle();
    const double phi1 = random_angle();
    const double lambda1 = random_angle();
    circuit->addInstruction(provider->createInstruction(
        "U", {0},
        std::vector<xacc::InstructionParameter>{theta1, phi1, lambda1}));
    const double theta2 = random_angle();
    const double phi2 = random_angle();
    const double lambda2 = random_angle();
    circuit->addInstruction(provider->createInstruction(
        "U", {1},
        std::vector<xacc::InstructionParameter>{theta2, phi2, lambda2}));
    circuit->addInstruction(provider->createInstruction("CNOT", {0, 1}));
    circuit->addInstruction(provider->createInstruction("H", {0}));
    circuit->addInstruction(provider->createInstruction("H", {1}));
    auto [mirror_cir, expected_result] = qcor::MirrorCircuitValidator::createMirrorCircuit(
        std::make_shared<qcor::CompositeInstruction>(circuit));
    const std::string expectedBitString =
        std::to_string(expected_result[0]) + std::to_string(expected_result[1]);
    // std::cout << "HOWDY: \n" << mirror_cir->toString() << "\n";
    // std::cout << "Expected bitstring: " << expectedBitString << "\n";
    auto mirror_circuit = provider->createComposite("test_mirror");
    mirror_circuit->addInstructions(mirror_cir->getInstructions());
    mirror_circuit->addInstruction(provider->createInstruction("Measure", {0}));
    mirror_circuit->addInstruction(provider->createInstruction("Measure", {1}));
    auto mc_buffer = xacc::qalloc(2);
    accelerator->execute(mc_buffer, mirror_circuit);
    //mc_buffer->print();
    EXPECT_EQ(mc_buffer->getMeasurementCounts().size(), 1);
    EXPECT_EQ(mc_buffer->getMeasurementCounts()[expectedBitString], 1024);
    allBitStrings.emplace(expectedBitString);
  }
  // Cover both cases (randomized Pauli worked)
  EXPECT_EQ(allBitStrings.size(), 4);
}

TEST(MirrorCircuitTester, checkDeuteron) {
  auto provider = xacc::getIRProvider("quantum");
  constexpr int NUM_TESTS = 100;
  auto accelerator = xacc::getAccelerator("qpp", {{"shots", 1024}});
  std::set<std::string> allBitStrings;
  for (int i = 0; i < NUM_TESTS; ++i) {
    auto circuit =
        provider->createComposite(std::string("test") + std::to_string(i));
    circuit->addInstruction(provider->createInstruction("X", {0}));
    circuit->addInstruction(provider->createInstruction(
        "Ry", {1}, std::vector<xacc::InstructionParameter>{random_angle()}));
    circuit->addInstruction(provider->createInstruction("CNOT", {1, 0}));
    auto [mirror_cir, expected_result] = qcor::MirrorCircuitValidator::createMirrorCircuit(
        std::make_shared<qcor::CompositeInstruction>(circuit));
    // std::cout << "Expected: " << expected_result[0] << expected_result[1] << "\n";
    // std::cout << "HOWDY: \n" << mirror_cir->toString() << "\n";
    auto mirror_circuit = provider->createComposite("test_mirror");
    mirror_circuit->addInstructions(mirror_cir->getInstructions());
    mirror_circuit->addInstruction(provider->createInstruction("Measure", {0}));
    mirror_circuit->addInstruction(provider->createInstruction("Measure", {1}));
    auto mc_buffer = xacc::qalloc(2);
    accelerator->execute(mc_buffer, mirror_circuit);
    // mc_buffer->print();
    const std::string expectedBitString =
        std::to_string(expected_result[0]) + std::to_string(expected_result[1]);
    EXPECT_EQ(mc_buffer->getMeasurementCounts().size(), 1);
    EXPECT_EQ(mc_buffer->getMeasurementCounts()[expectedBitString], 1024);
    allBitStrings.emplace(expectedBitString);
  }
  // We should have seen all 4 possible cases with that number of randomized
  // Pauli runs
  EXPECT_EQ(allBitStrings.size(), 4);
}

TEST(MirrorCircuitTester, checkNoise) {
  const std::string msb_noise_model =
      R"({"gate_noise": [{"gate_name": "CNOT", "register_location": ["0", "1"], "noise_channels": [{"matrix": [[[[0.99498743710662, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.99498743710662, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.99498743710662, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.99498743710662, 0.0]]], [[[0.0, 0.0], [0.05773502691896258, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.05773502691896258, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.05773502691896258, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.05773502691896258, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, -0.05773502691896258], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.05773502691896258], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, -0.05773502691896258]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.05773502691896258], [0.0, 0.0]]], [[[0.05773502691896258, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-0.05773502691896258, 0.0], [0.0, 0.0], [-0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.05773502691896258, 0.0], [0.0, 0.0]], [[0.0, 0.0], [-0.0, 0.0], [0.0, 0.0], [-0.05773502691896258, 0.0]]]]}]}], "bit_order": "MSB"})";

  auto noiseModel = xacc::getService<xacc::NoiseModel>("json");
  noiseModel->initialize({{"noise-model", msb_noise_model}});
  const std::string ibmNoiseJson = noiseModel->toJson();
  auto accelerator =
      xacc::getAccelerator("aer", {{"noise-model", ibmNoiseJson}});
  auto provider = xacc::getIRProvider("quantum");
  auto circuit = provider->createComposite(std::string("test"));
  circuit->addInstruction(provider->createInstruction("H", {0}));
  circuit->addInstruction(provider->createInstruction("H", {1}));
  circuit->addInstruction(provider->createInstruction("CNOT", {0, 1}));
  // xacc::set_verbose(true);
  auto validator = xacc::getService<qcor::BackendValidator>("mirror-rb");
  {
    auto [success, data] = validator->validate(
        accelerator, std::make_shared<qcor::CompositeInstruction>(circuit));
    EXPECT_TRUE(success);
  }
  {
    // Validate with a tighter limit (0.01 == 1%)
    auto [success, data] = validator->validate(
        accelerator, std::make_shared<qcor::CompositeInstruction>(circuit),
        {{"epsilon", 0.01}});
    EXPECT_FALSE(success);
  }
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
