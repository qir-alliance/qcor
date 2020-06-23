#include "qcor.hpp"
#include "qrt.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_quantum_gate_api.hpp"
#include "xacc_service.hpp"
#include <gtest/gtest.h>

using namespace xacc;
const std::string rucc = R"rucc(__qpu__ void f(qbit q, double t0) {
    X(q[0]);
    Ry(q[1],t0);
    CNOT(q[1],q[0]);
})rucc";

const std::string rucc_vec =
    R"rucc(__qpu__ void f_vec(qbit q, std::vector<double> x) {
    X(q[0]);
    Ry(q[1],x[0]);
    CNOT(q[1],q[0]);
})rucc";

TEST(QCORTester, checkTaskInitiate) {

  xacc::internal_compiler::compiler_InitializeXACC("qpp");

  auto buffer = qalloc(4);

  auto ruccsd = xacc::getService<xacc::Compiler>("xasm")
                    ->compile(rucc, nullptr)
                    ->getComposite("f");

  auto ruccsd_vec = xacc::getService<xacc::Compiler>("xasm")
                        ->compile(rucc_vec, nullptr)
                        ->getComposite("f_vec");

  auto optimizer = qcor::createOptimizer("nlopt");
  std::shared_ptr<Observable> observable = xacc::quantum::getObservable(
      "pauli",
      std::string("5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1"));

  // Create the ObjectiveFunction, here we want to run VQE
  // need to provide ansatz and the Observable
  auto objective = xacc::getService<qcor::ObjectiveFunction>("vqe");
  objective->initialize(observable.get(), ruccsd);
  objective->set_qreg(buffer);

  auto args_translation =
      qcor::TranslationFunctor<xacc::internal_compiler::qreg, double>(
          [&](const std::vector<double> x) {
            return std::make_tuple(buffer, x[0]);
          });

  auto handle = qcor::taskInitiate(objective, optimizer, args_translation, 1);
  auto results = qcor::sync(handle);
  EXPECT_NEAR(-1.748865, results.opt_val, 1e-4);

  handle = qcor::taskInitiate(
      objective, optimizer,
      [&](const std::vector<double> x, std::vector<double> &dx) {
        return (*objective)(buffer, x[0]);
      },
      1);

  auto results2 = qcor::sync(handle);
  EXPECT_NEAR(-1.748865, results2.opt_val, 1e-4);

  qcor::OptFunction ff(
      [&](const std::vector<double> x, std::vector<double> &dx) {
        return (*objective)(buffer, x[0]);
      },
      1);
  handle = qcor::taskInitiate(objective, optimizer, ff);

  auto results3 = qcor::sync(handle);
  EXPECT_NEAR(-1.748865, results3.opt_val, 1e-4);

  auto l = [&](const std::vector<double> x, std::vector<double> &dx) {
    return (*objective)(buffer, x[0]);
  };
  handle = qcor::taskInitiate(objective, optimizer, l, 1);

  auto results4 = qcor::sync(handle);
  EXPECT_NEAR(-1.748865, results4.opt_val, 1e-4);

  auto objective_vec = xacc::getService<qcor::ObjectiveFunction>("vqe");
  objective->initialize(observable.get(), ruccsd_vec);

  objective->set_qreg(buffer);

  auto args_translation_vec = qcor::TranslationFunctor<
      xacc::internal_compiler::qreg, std::vector<double>>(
      [&](const std::vector<double> x) { return std::make_tuple(buffer, x); });

  auto handle2 =
      qcor::taskInitiate(objective_vec, optimizer, args_translation_vec, 1);
  auto results5 = qcor::sync(handle2);
  EXPECT_NEAR(-1.748865, results5.opt_val, 1e-4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
