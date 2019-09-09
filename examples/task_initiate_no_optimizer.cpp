#include "qcor.hpp"

template <typename T> std::vector<T> linspace(T a, T b, size_t N) {
  T h = (b - a) / static_cast<T>(N - 1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
    *x = val;
  return xs;
}

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto ansatz = [&](qbit q, std::vector<double> t) {
    X(q[0]);
    Ry(q[1], t[0]);
    CNOT(q[1], q[0]);
  };

  auto observable =
      qcor::getObservable("pauli", std::string("5.907 - 2.1433 X0X1 "
                                               "- 2.1433 Y0Y1"
                                               "+ .21829 Z0 - 6.125 Z1"));

  std::vector<double> all_params =
      linspace(-xacc::constants::pi, xacc::constants::pi, 10);
  for (auto p : all_params) {
    std::vector<double> pv{p};
    auto handle = qcor::taskInitiate(ansatz, "vqe", observable,
                                     pv);
    auto results = qcor::sync(handle);

    // std::cout << results->getInformation("opt-val").as<double>() << "\n";
  }
  qcor::Finalize();
}
