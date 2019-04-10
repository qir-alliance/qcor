#include <gtest/gtest.h>

#include "ChemistryObservable.hpp"
#include "XACC.hpp"
#include "qcor.hpp"
#include "xacc_service.hpp"

using namespace qcor;
using namespace qcor::observable;

TEST(VQETester, checkSimple) {

  ChemistryObservable observable;
  std::string geom = R"geom(2

H          0.00000        0.00000        0.00000
H          0.00000        0.00000        0.7474
)geom";

  observable.fromOptions(
      {{"basis", "sto-3g"},
       {"geometry", "2\ntest geom\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7474\n"}});
}

int main(int argc, char **argv) {
  qcor::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
