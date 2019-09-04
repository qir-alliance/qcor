#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer = qcor::getOptimizer(
      "nlopt", {{"nlopt-optimizer", "cobyla"}, {"nlopt-maxeval", 1000}});

  auto geom = R"geom(2

H          0.00000        0.00000        0.00000
H          0.00000        0.00000        0.7474)geom";

  auto op = qcor::getObservable("chemistry",
                                {std::make_pair("basis", "sto-3g"), std::make_pair("geometry", geom)});

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](qbit q, double x) {
          X(q[0]);
          X(q[2]);
          exp_i_theta(q, x, {{"pauli", "Y0 X1 X2 X3"}});
        },
        op, optimizer, 0.0);
  });

  auto results = future.get();
  auto energy = mpark::get<double>(results->getInformation("opt-val"));
}
