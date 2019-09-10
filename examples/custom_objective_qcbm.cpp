#include "qcor.hpp"

class QCBM : public qcor::ObjectiveFunction {
protected:
  // Example Bars and Stripes PDF
  std::vector<double> target{
      0.16666667, 0.,         0., 0.16666667, 0., 0.16666667, 0.,        0., 0.,
      0.,         0.16666667, 0., 0.16666667, 0., 0.,         0.16666667};

  // Helper function
  double entropy(const std::vector<double> p, const std::vector<double> q) {
    double sum = 0.0;
    for (int i = 0; i < p.size(); i++) {
      if (std::fabs(p[i]) > 1e-12) {
        sum += p[i] * std::log(p[i] / q[i]);
      }
    }
    return sum;
  }

public:
  // All ObjectiveFunctions must implement this method.
  // QCOR will take your ObjectiveFunction and inject it
  // with the Observable (observable), Kernel (kernel),
  // and Global ResultsBuffer (buffer). You can use those as
  // you wish to construct your function. Also, note you have
  // reference to the backend QPU (backend) that this
  // file was compiled for.
  double operator()(const std::vector<double> &x) override {
    // Evaluate the ansatz at the given parameters.
    auto evaledKernel = kernel->operator()(x);
    // Observe the ansatz (appends measurements). We
    // should only have one measured kernel
    auto observed = observable->observe(evaledKernel)[0];

    // Create a temp buffer to store results
    auto tmpBuffer = qcor::qalloc(buffer->size());
    // Execute
    backend->execute(tmpBuffer, observed);
    // Get the counts
    auto counts = tmpBuffer->getMeasurementCounts();
    int shots = 0;
    for (auto &x : counts) {
      shots += x.second;
    }

    // Compute the probability distribution
    std::vector<double> q(target.size()); // all zeros
    for (auto &x : counts) {
      int idx = std::stoi(x.first, nullptr, 2);
      q[idx] = (double) x.second / shots;
    }
    // get M=1/2(P+Q)
    std::vector<double> m(target.size());
    for (int i = 0; i < m.size(); i++)
      m[i] = .5 * (target[i] + q[i]);

    auto js = 0.5 * (entropy(target, m) + entropy(q, m));
    std::cout << "JS: " << js <<  "\n";
    return js;
  }
};

int main(int argc, char **argv) {

  // Initialize QCOR
  qcor::Initialize(argc, argv);

  // Create our custom ObjectiveFunction
  // as a shared_ptr, this is the QCBM work
  // from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.062323
  auto qcbm = std::make_shared<QCBM>();

  // Define a few of the CNOT couplings for the
  // HWE ansatz, the paper defines dc3 and dc4 amongst others
  std::vector<std::pair<int, int>> dc3{{0, 1}, {0,2}, {0, 3}};
  std::vector<std::pair<int, int>> dc4{{0, 1}, {0,2}, {1, 3}, {2,3}};

  // Create the quantum kernel, a standard lambda
  // that uses the HWE circuit generator
  auto ansatz = [&](qbit q, std::vector<double> x) {
    hwe(q, x, {{"nq", 4}, {"layers", 1}, {"coupling", dc4}});
  };

  // Create the NLOpt cobyla optimizer
  // FIXME implement ADAM optimizer from mlpack
  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 500)});

  // Call taskInitiate with the objective function
  // and optimizer, default initial params (all zeros)
  // No observable implies QCOR will measure in
  // computational basis
  auto handle =
      qcor::taskInitiate(ansatz, qcbm, optimizer, std::vector<double>{});

  // We have the handle, its an async call,
  // maybe go do other work...

  // Get the results
  auto results = qcor::sync(handle);

  auto optJS = results->getInformation("opt-val").as<double>();
  auto params = results->getInformation("opt-params").as<std::vector<double>>();

  std::cout << "Opt JS = " << optJS << "\n";
  std::cout << "At parameters: "
            << params
            << "\n";

  // Finalize the framewokr
  qcor::Finalize();
}
