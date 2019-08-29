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

  // Create an unmeasured ansatz
  auto ansatz = [&](qbit q, std::vector<double> t) {
    X(q[0]);
    Ry(q[1], t[0]);
    CNOT(q[1], q[0]);
  };

  // Manually add Measure calls to the
  // existing ansatz
  auto Z0 = [&](qbit q, std::vector<double> t) {
    Measure(q[0]);
  };
  auto ansatzZ0 = qcor::add(ansatz, Z0, std::vector<double>{});

  // Now lets compute <Z0>(theta)
  std::vector<double> all_params =
      linspace(-xacc::constants::pi, xacc::constants::pi, 10);
  for (auto &p : all_params) {
    auto buffer = xacc::qalloc(2);
    ansatzZ0(buffer, std::vector<double>{p});
    std::cout << "<Z0Z1> = " << buffer->getInformation("exp-val-z").as<double>()
              << "\n";
  }

  qcor::Finalize();
}
