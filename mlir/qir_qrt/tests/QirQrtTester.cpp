#include "gtest/gtest.h"
#include <xacc.hpp>
#include "qir-qrt.hpp"
#include "qrt.hpp"
#include "xacc_service.hpp"

TEST(QirQrtTester, checkSimple) {
  ::quantum::qrt_impl = xacc::getService<::quantum::QuantumRuntime>("nisq");
  ::quantum::qrt_impl->initialize("empty");
  int nbQubits = 5;
  auto arrayPtr = __quantum__rt__qubit_allocate_array(nbQubits);
  EXPECT_EQ(arrayPtr->size(), nbQubits);
  for (int i = 0; i < nbQubits; ++i) {
    Qubit *qbit = *(reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(arrayPtr, i)));

    __quantum__qis__h(qbit);
  }

  for (int i = 0; i < nbQubits - 1; ++i) {
    Qubit *src = *(reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(arrayPtr, i)));
    Qubit *tgt = *(reinterpret_cast<Qubit **>(
        __quantum__rt__array_get_element_ptr_1d(arrayPtr, i + 1)));

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

int main(int argc, char **argv) {
  __quantum__rt__initialize(argc, (int8_t**)(argv));
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  __quantum__rt__finalize();
  return ret;
}