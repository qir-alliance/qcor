#include "gtest/gtest.h"
#include <xacc.hpp>
#include "qir-qrt.hpp"

TEST(QirQrtTester, checkSimple) {
  std::cout << "HOWDY\n";
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
}

int main(int argc, char **argv) {
  __quantum__rt__initialize(argc, (int8_t**)(argv));
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  __quantum__rt__finalize();
  return ret;
}