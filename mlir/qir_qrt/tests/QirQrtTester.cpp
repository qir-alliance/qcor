#if __GNUC__ >= 5
// Disable GCC 5's -Wsuggest-override and -Wsign-compare warnings in gtest
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsuggest-override"
# pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#include "gtest/gtest.h"
#include <xacc.hpp>
#include "qir-qrt.hpp"
#include "qrt.hpp"
#include "xacc_service.hpp"

namespace {
constexpr int nbQubits = 5;
static Array *globalQubitArrayPtr = nullptr;
} // namespace

TEST(QirQrtTester, checkSimple) {
  ::quantum::qrt_impl = xacc::getService<::quantum::QuantumRuntime>("nisq");
  ::quantum::qrt_impl->initialize("empty");
  if (!globalQubitArrayPtr) {
    globalQubitArrayPtr = __quantum__rt__qubit_allocate_array(nbQubits);
  }
  EXPECT_EQ(globalQubitArrayPtr->size(), nbQubits);
  for (int i = 0; i < nbQubits; ++i) {
    Qubit *qbit = *(reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(globalQubitArrayPtr, i)));

    __quantum__qis__h(qbit);
  }

  for (int i = 0; i < nbQubits - 1; ++i) {
    Qubit *src = *(reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(globalQubitArrayPtr, i)));
    Qubit *tgt = *(reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(globalQubitArrayPtr, i + 1)));

    __quantum__qis__cnot(src, tgt);
  }
  std::cout << "HOWDY:\n"
            << ::quantum::qrt_impl->get_current_program()->toString() << "\n";
  EXPECT_EQ(::quantum::qrt_impl->get_current_program()->nInstructions(), 9);
  for (int i = 0; i < nbQubits; ++i) {
    auto inst = ::quantum::qrt_impl->get_current_program()->getInstruction(i);
    EXPECT_EQ(inst->name(), "H");
    EXPECT_EQ(inst->bits().size(), 1);
    EXPECT_EQ(inst->bits()[0], i);
  }
}

TEST(QirQrtTester, checkArray) {
  ::quantum::qrt_impl = xacc::getService<::quantum::QuantumRuntime>("nisq");
  ::quantum::qrt_impl->initialize("empty");
  if (!globalQubitArrayPtr) {
    globalQubitArrayPtr = __quantum__rt__qubit_allocate_array(nbQubits);
  }
  EXPECT_EQ(globalQubitArrayPtr->size(), nbQubits);
  // Create an alias array of 3 qubits
  const std::vector<int> qubitsToCopy {0, 2, 4};
  auto aliasQubitArray = __quantum__rt__array_create_1d(sizeof(Qubit *), qubitsToCopy.size());
  EXPECT_EQ(aliasQubitArray->size(), qubitsToCopy.size());
  for (int i = 0; i < qubitsToCopy.size(); ++i) {
    auto src_loc =
        __quantum__rt__array_get_element_ptr_1d(globalQubitArrayPtr, qubitsToCopy[i]);
    auto target_loc =
        __quantum__rt__array_get_element_ptr_1d(aliasQubitArray, i);
    memcpy(target_loc, src_loc, sizeof(Qubit *));
  }

  for (int i = 0; i < aliasQubitArray->size(); ++i) {
    Qubit *qb = *(reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(aliasQubitArray, i)));
    __quantum__qis__h(qb);
  }
  std::cout << "HOWDY:\n"
            << ::quantum::qrt_impl->get_current_program()->toString() << "\n";
  EXPECT_EQ(::quantum::qrt_impl->get_current_program()->nInstructions(), qubitsToCopy.size());
  // Check that the Hadamard gates are applied to the correct qubits {0, 2, 4}
  for (int i = 0; i < qubitsToCopy.size(); ++i) {
    auto inst = ::quantum::qrt_impl->get_current_program()->getInstruction(i);
    EXPECT_EQ(inst->name(), "H");
    EXPECT_EQ(inst->bits().size(), 1);
    EXPECT_EQ(inst->bits()[0], qubitsToCopy[i]);
  }
}

#if __GNUC__ >= 5
# pragma GCC diagnostic pop
#endif

int main(int argc, char **argv) {
  __quantum__rt__initialize(argc, (int8_t**)(argv));
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  __quantum__rt__finalize();
  return ret;
}