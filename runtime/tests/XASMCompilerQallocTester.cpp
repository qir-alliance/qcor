/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "gtest/gtest.h"

#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_quantum_gate_api.hpp"
#include <qalloc>

TEST(XASMCompilerQallocTester, checkSimpleFor) {

  auto compiler = xacc::getCompiler("xasm");
  auto IR =
      compiler->compile(R"(__qpu__ void testFor(qbit q, std::vector<double> x) {
  for (int i = 0; i < 5; i++) {
     H(q[i]);
  }
  for (int i = 0; i < 2; i++) {
      Rz(q[i], x[i]);
  }
})");
  std::cout << "KERNEL\n" << IR->getComposites()[0]->toString() << "\n";

  xacc::internal_compiler::qreg q(5);
  auto tt = IR->getComposites()[0];
  tt->updateRuntimeArguments(q, std::vector<double>{1.2, 3.4});
  std::cout << "EVALED NEW WAY:\n" << tt->toString() << "\n";

  IR = compiler->compile(
      R"(__qpu__ void testFor2(qbit q, std::vector<double> x) {
  for (int i = 0; i < 5; i++) {
     H(q[i]);
     Rx(q[i], x[i]);
     CX(q[0], q[i]);
  }

  for (int i = 0; i < 3; i++) {
      CX(q[i], q[i+1]);
  }
  Rz(q[3], 0.22);

  for (int i = 3; i > 0; i--) {
      CX(q[i-1],q[i]);
  }
})");
  EXPECT_EQ(1, IR->getComposites().size());
  std::cout << "KERNEL\n" << IR->getComposites()[0]->toString() << "\n";
  for (auto ii : IR->getComposites()[0]->getVariables())
    std::cout << ii << "\n";
  EXPECT_EQ(22, IR->getComposites()[0]->nInstructions());

  IR->getComposites()[0]->updateRuntimeArguments(
      q, std::vector<double>{1, 2, 3, 4, 5});
  std::cout << "KERNEL\n" << IR->getComposites()[0]->toString() << "\n";
}
TEST(XASMCompilerQallocTester, checkHWEFor) {

  auto compiler = xacc::getCompiler("xasm");
  auto IR = compiler->compile(R"([&](qbit q, std::vector<double> x) {
    for (int i = 0; i < 2; i++) {
        Rx(q[i],x[i]);
        Rz(q[i],x[2+i]);
    }
    CX(q[1],q[0]);
    for (int i = 0; i < 2; i++) {
        Rx(q[i], x[i+4]);
        Rz(q[i], x[i+4+2]);
        Rx(q[i], x[i+4+4]);
    }
  })");
  EXPECT_EQ(1, IR->getComposites().size());
  std::cout << "KERNEL\n" << IR->getComposites()[0]->toString() << "\n";
  for (auto ii : IR->getComposites()[0]->getVariables())
    std::cout << ii << "\n";
  EXPECT_EQ(11, IR->getComposites()[0]->nInstructions());

  xacc::internal_compiler::qreg q(2);

  IR->getComposites()[0]->updateRuntimeArguments(
      q, std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::cout << "KERNEL\n" << IR->getComposites()[0]->toString() << "\n";
}

TEST(XASMCompilerQallocTester, checkIRV3) {
  //   auto v = xacc::qalloc(1);
  //   v->setName("v");
  //   xacc::storeBuffer(v);

  //   auto v = xacc::internal_compiler::qalloc(1);
  xacc::internal_compiler::qreg v(1);

  auto H = xacc::quantum::getObservable("pauli", std::string("X0 Y1 + Y0 X1"));

  auto compiler = xacc::getCompiler("xasm");
  auto IR = compiler->compile(
      R"(
   __qpu__ void foo_test (qbit v, double x, double y, double z, std::shared_ptr<Observable> H) {
     Rx(v[0], x);
     U(v[0], x, y, z);
     exp_i_theta(v, x, H);
   }
   )");

  auto foo_test = IR->getComposite("foo_test");

  std::cout << foo_test->toString() << "\n";

  for (auto &val : {2.2, 2.3, 2.4, 2.5}) {
    foo_test->updateRuntimeArguments(v, val, 3.3, 4.4, H);

    std::cout << foo_test->toString() << "\n\n";
  }
}

TEST(XASMCompilerQallocTester, checkIRV3Vector) {
  //   auto v = xacc::qalloc(1);
  //   v->setName("v");
  //   xacc::storeBuffer(v);

  //   auto v = xacc::internal_compiler::qalloc(1);
  xacc::internal_compiler::qreg v(1);

  auto H = xacc::quantum::getObservable("pauli", std::string("X0 Y1 + Y0 X1"));

  auto compiler = xacc::getCompiler("xasm");
  auto IR = compiler->compile(
      R"(
   __qpu__ void foo_test2 (qbit v, std::vector<double> x, std::shared_ptr<Observable> H) {
     Rx(v[0], x[0]);
     U(v[0], x[0], x[1], x[2]);
     exp_i_theta(v, x[1], H);
   }
   )");

  auto foo_test = IR->getComposite("foo_test2");

  std::cout << foo_test->toString() << "\n";

  for (auto &val : {2.2, 2.3, 2.4, 2.5}) {
    foo_test->updateRuntimeArguments(v, std::vector<double>{val, 3.3, 4.4}, H);

    std::cout << foo_test->toString() << "\n\n";
  }

  IR = compiler->compile(
      R"(
  __qpu__ void ansatz2(qreg q, std::vector<double> theta) {
  X(q[0]);
  Ry(q[1], theta[0]);
  CX(q[1],q[0]);
}
)");

  auto test = IR->getComposites()[0];
  std::cout << " HELLO: " << test->toString() << "\n";
  test->updateRuntimeArguments(v, std::vector<double>{.48});
  std::cout << " HELLO: " << test->toString() << "\n";
}

TEST(XASMCompilerQallocTester, checkIRV3Expression) {
  //   auto v = xacc::qalloc(1);
  //   v->setName("v");
  //   xacc::storeBuffer(v);

  //   auto v = xacc::internal_compiler::qalloc(1);
  xacc::internal_compiler::qreg v(1);

  auto compiler = xacc::getCompiler("xasm");
  auto IR = compiler->compile(
      R"(
   __qpu__ void foo_test3 (qbit v, double x) {
     Rx(v[0], 2.2*x+pi);
   }
   )");

  auto foo_test = IR->getComposite("foo_test3");

  std::cout << foo_test->toString() << "\n";

  for (auto &val : {2.2, 2.3, 2.4, 2.5}) {
    foo_test->updateRuntimeArguments(v, val);

    std::cout << foo_test->toString() << "\n\n";
  }
}

TEST(XASMCompilerQallocTester, checkAnnealInstructions) {
  xacc::internal_compiler::qreg v(1);

  auto compiler = xacc::getCompiler("xasm");
  auto IR = compiler->compile(
      R"(
   __qpu__ void anneal_test (qbit v, double x, double y) {
       QMI(v[0], x);
       QMI(v[1], y);
       QMI(v[0], v[1], .2345);
   }
   )");

  auto foo_test = IR->getComposite("anneal_test");

  std::cout << foo_test->toString() << "\n";

  for (auto &val : {2.2, 2.3, 2.4, 2.5}) {
    foo_test->updateRuntimeArguments(v, val, 3.3);

    std::cout << foo_test->toString() << "\n\n";
  }

  IR = compiler->compile(
      R"(
  __qpu__ void ansatz222(qreg v, std::vector<double> x) {
    QMI(v[0], x[0]);
    QMI(v[1], x[1]);
    QMI(v[0], v[1], x[2]);
}
)");

  auto test = IR->getComposites()[0];
  std::cout << " HELLO: " << test->toString() << "\n";
  test->updateRuntimeArguments(v, std::vector<double>{.48, .58, .68});
  std::cout << " HELLO: " << test->toString() << "\n";

  IR = compiler->compile(
      R"(
  __qpu__ void rbm_test(qreg v, std::vector<double> x, int nv, int nh) {
    rbm(v, x, nv, nh);
}
)");
  test = IR->getComposites()[0];

  for (int i = 1; i < 4; i++) {
    //   std::cout << " HELLO: " << test->toString() << "\n";
    test->updateRuntimeArguments(v, std::vector<double>(i * i + i + i), i, i);
    std::cout << " HELLO:\n" << test->toString() << "\n";
  }
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  xacc::set_verbose(true);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
