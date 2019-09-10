#ifndef RUNTIME_QCOR_HPP_
#define RUNTIME_QCOR_HPP_

#include "Observable.hpp"
#include "heterogeneous.hpp"
#include "xacc.hpp"
#include <future>
#include <vector>

using namespace xacc;

// The QCOR Public API provides the following functions:
// (1) Initialize - Initialize the QCOR runtime library. This
//                  is required to be called once before all invocations
//                  of qcor library calls. It optionally takes as input the
//                  program command line arguments. The primary function of
//                  this call is to initialize the XACC framework, specifically
//                  its plugin/service registry system.
//
// (2) Finalize - Finalize the QCOR runtime library. This should be called at
// the
//                end of a QCOR-enabled program to cleanup and finalize QCOR.
//                Specifically, this Finalizes the XACC framework which cleans
//                up the provided service pointers.
//
// (3) submit - This function enables job or task submission to QCOR for
// asynchronous
//              execution. Tasks are functor-like objects (lambdas for example)
//              that take a single argument - a reference to a
//              qcor::qpu_handler. Tasks can use this qpu_handler to execute
//              desired objective functions or hybrid algorithms. This function
//              returns a C++ future object wrapping the to-be-populated
//              AcceleratorBuffer containing execution results (counts, and a
//              heterogeneous map of execution data). This method is overloaded
//              to accept an already created AcceleratorBuffer.
//
// (4) getOptimizer - This function returns a concrete Optimizer implementation
// (such as the nlopt
//                    optimizer). It is overloaded to optionally take a
//                    heterogeneous map of optimizer options. This method
//                    leverages the XACC service registry to get reference to
//                    the desired optimizer service implementation.
//
// (5) getObservable - This function returns a concrete Observable
// implementation instance.
//                     This Observable dictates measurements on an unmeasured
//                     quantum kernel. It is overloaded to enable creation of an
//                     Observable from a particular string representation or a
//                     heterogeneous map of options.
//
// (6) add - This function adds to quantum kernels together, i.e. appends the
// instructions
//           of the second one to the first. This returns a new quantum kernel
//           representing this addition.
//
// QCOR C++ Quantum Kernels:
// -------------------------
// Quantum Kernels in QCOR are represented as standard C++ functors, which may
// be free functions or lambdas. These functors must take as its first argument
// a reference to a register of qubits (of qbit type). The following arguments
// can be runtime parameters (circuit parameters) that are either of double
// value, or a std::vector<double>. A prototypical example looks like this:
//
// auto kernel = [&](qbit q, std::vector<double> p) {
//  X(q[0]);
//  Ry(q[1], p[0]);
//  CX(q[1], q[0]);
//  Measure(q[0]);
// };
//
// Note that the default quantum language leveraged here is the XACC xasm
// language.
//
// If measure instructions are specified, one may execute quantum kernels just
// as you would any other classical functor:
//
// auto q = qcor::qalloc(2);
// kernel(q, std::vector<double>{2.2});
// auto counts = q->getMeasurementCounts();
//
// This form of quantum kernel execution is a serial. To enact asynchronous
// execution please leverage the qcor::submit or qcor::taskInitiate API call.

namespace qcor {

// The ResultBuffer concept from the QCOR specification
// provides a means for connecting clients/programmers with
// execution results.
using ResultBuffer = xacc::qbit;

// The Handle concept fromthe QCOR specification gives
// programmers a data structure to hold on to while
// asynchronous execution is on-going, they can use
// this handle to query the results using the qcor::sync call
using Handle = std::future<ResultBuffer>;

class ObjectiveFunction
    : public std::function<double(const std::vector<double> &)> {
protected:
  std::shared_ptr<Observable> observable;
  std::shared_ptr<CompositeInstruction> kernel;
  std::vector<std::shared_ptr<CompositeInstruction>> kernels;
  std::shared_ptr<Accelerator> backend;
  std::shared_ptr<AcceleratorBuffer> buffer;

public:
  virtual double operator()(const std::vector<double> &params) = 0;
  void initialize(std::shared_ptr<Observable> obs,
                  std::shared_ptr<CompositeInstruction> k, std::shared_ptr<AcceleratorBuffer> b) {
    observable = obs;
    kernel = k;
    kernels = obs->observe(k);
    backend = xacc::getAccelerator(kernel->accelerator_signature());
    buffer = b;
  }
};

// The QCOR submit API requires tasks be submitted as
// functors that take a qpu_handler reference and return void.
class qpu_handler;
using HandlerLambda = std::function<void(qpu_handler &)>;

namespace __internal {

extern bool executeKernel;

void switchDefaultKernelExecution(bool execute);

template <typename QuantumKernel, typename... Args>
std::shared_ptr<CompositeInstruction>
getCompositeInstruction(QuantumKernel &kernel, Args... args) {
  qcor::__internal::switchDefaultKernelExecution(false);

  auto qb = xacc::qalloc(1000);
  auto persisted_function = kernel(qb, args...);

  qcor::__internal::switchDefaultKernelExecution(true);

  auto function = xacc::getIRProvider("quantum")->createComposite("f");
  std::istringstream iss(persisted_function);
  function->load(iss);
  return function;
}

void updateMap(xacc::HeterogeneousMap &m, std::vector<double> &values);
void updateMap(xacc::HeterogeneousMap &m, std::vector<double> &&values);
void updateMap(xacc::HeterogeneousMap &m, double value);

void constructInitialParameters(xacc::HeterogeneousMap &m);
template <typename First, typename... Rest>
void constructInitialParameters(xacc::HeterogeneousMap &m, First firstArg,
                                Rest... rest) {
  updateMap(m, firstArg);
  constructInitialParameters(m, rest...);
}

template <typename... Args>
std::vector<double>
parametersFromVariadic(const std::shared_ptr<CompositeInstruction> function,
                       Args... args) {
  HeterogeneousMap tmp;
  tmp.insert("initial-parameters", std::vector<double>{});
  tmp.insert("__internal_n_vars", function->nVariables());
  __internal::constructInitialParameters(tmp, args...);
  return tmp.get<std::vector<double>>("initial-parameters");
}
} // namespace __internal

void Initialize();
void Initialize(int argc, char **argv);
void Initialize(std::vector<std::string> argv);
void Finalize();

ResultBuffer qalloc(const std::size_t nBits);
ResultBuffer qalloc();

Handle submit(HandlerLambda &&lambda);
Handle submit(HandlerLambda &&lambda,
              std::shared_ptr<xacc::AcceleratorBuffer> buffer);

ResultBuffer sync(Handle &handle);

std::shared_ptr<xacc::Optimizer> getOptimizer(const std::string &name);
std::shared_ptr<xacc::Optimizer>
getOptimizer(const std::string &name, const xacc::HeterogeneousMap &&options);

std::shared_ptr<xacc::Observable>
getObservable(const std::string &type, const std::string &representation);
std::shared_ptr<xacc::Observable> getObservable();
std::shared_ptr<xacc::Observable>
getObservable(const std::string &representation);
std::shared_ptr<xacc::Observable>
getObservable(const std::string &type, const xacc::HeterogeneousMap &&options);

template <typename QuantumKernelA, typename QuantumKernelB, typename... Args>
std::function<std::string(qbit, Args...)>
add(QuantumKernelA &qka, QuantumKernelB &qkb, Args... args) {

  auto function1 = qcor::__internal::getCompositeInstruction(qka, args...);
  auto function2 = qcor::__internal::getCompositeInstruction(qkb, args...);

  for (auto &inst : function2->getInstructions()) {
    function1->addInstruction(inst);
  }

  return [=](qbit q, Args... args) {
    if (qcor::__internal::executeKernel) {
      HeterogeneousMap tmp;
      tmp.insert("initial-parameters", std::vector<double>{});
      qcor::__internal::constructInitialParameters(tmp, args...);
      std::vector<double> params =
          tmp.get<std::vector<double>>("initial-parameters");
      auto accelerator =
          xacc::getAccelerator(function1->accelerator_signature());
      auto evaled = function1->operator()(params);
      accelerator->execute(q, evaled);
    }
    std::stringstream ss;
    function1->persist(ss);
    return ss.str();
  };
}

// QCOR QPU Handler:
// The qpu_handler class is a convenience class for providing common algorithmic
// primitives or objective functions for optimization. For example, the
// variational quantum eigensolver algorithm is a commonly used hybrid
// algorithm, and the qpu_handler exposes a function for executing it with a
// given quantum kernel, observable, and optimizer. qpu_handler also exposes
// standard execute() methods for executing measured quantum kernels (kernels
// with measurement instructions specified). This function is overloaded for the
// execution of available XACC algorithms.
class qpu_handler {
protected:
  std::shared_ptr<xacc::AcceleratorBuffer> buffer;

public:
  qpu_handler() = default;
  qpu_handler(std::shared_ptr<xacc::AcceleratorBuffer> b) : buffer(b) {}

  ResultBuffer getResults() { return ResultBuffer(buffer); }

  template <typename QuantumKernel, typename... Args>
  void vqe(QuantumKernel &&kernel, std::shared_ptr<Observable> observable,
           std::shared_ptr<Optimizer> optimizer, Args... args) {
    vqe(kernel, observable, optimizer, args...);
  }

  template <typename QuantumKernel, typename... Args>
  void vqe(QuantumKernel &kernel, std::shared_ptr<Observable> observable,
           std::shared_ptr<Optimizer> optimizer, Args... args) {

    auto function = qcor::__internal::getCompositeInstruction(kernel, args...);

    auto nLogicalBits = function->nLogicalBits();
    auto accelerator = xacc::getAccelerator(function->accelerator_signature());

    if (!buffer) {
      buffer = xacc::qalloc(nLogicalBits);
    }

    auto parameters = __internal::parametersFromVariadic(function, args...);

    HeterogeneousMap options;
    options.insert("observable", observable);
    options.insert("ansatz", function);
    options.insert("optimizer", optimizer);
    options.insert("accelerator", accelerator);
    optimizer->appendOption("initial-parameters", parameters);

    auto vqeAlgo = xacc::getAlgorithm("vqe");
    bool success = vqeAlgo->initialize(options);
    if (!success) {
      xacc::error("Error initializing VQE algorithm.");
    }

    vqeAlgo->execute(buffer);
  }

  template <typename QuantumKernel, typename... Args>
  void execute(QuantumKernel &&kernel, Args... args) {
    auto function = qcor::__internal::getCompositeInstruction(kernel, args...);

    auto nLogicalBits = function->nLogicalBits();
    auto accelerator = xacc::getAccelerator(function->accelerator_signature());

    if (!buffer) {
      buffer = xacc::qalloc(nLogicalBits);
    }

    accelerator->execute(buffer, function);
  }

  template <typename QuantumKernel, typename... InitialArgs>
  void execute(const std::string &algorithm, QuantumKernel &&kernel,
               HeterogeneousMap &options, InitialArgs... args) {
    auto function = qcor::__internal::getCompositeInstruction(kernel, args...);

    auto nLogicalBits = function->nLogicalBits();
    auto accelerator = xacc::getAccelerator(function->accelerator_signature());
    if (!buffer) {
      buffer = xacc::qalloc(nLogicalBits);
    }

    auto parameters = __internal::parametersFromVariadic(function, args...);

    options.insert("ansatz", function);
    options.insert("accelerator", accelerator);

    std::shared_ptr<Algorithm> algo;
    if (algorithm == "vqe" &&
        !options.keyExists<std::shared_ptr<Optimizer>>("optimizer")) {
      options.insert("parameters", parameters);
      algo = xacc::getAlgorithm("vqe-energy");
    } else {
      algo = xacc::getAlgorithm(algorithm);
    }

    bool success = algo->initialize(options);
    if (!success) {
      xacc::error("Error initializing " + algorithm + " algorithm.");
    }

    algo->execute(buffer);
  }

  void execute(const std::string &algorithm, ResultBuffer buffer,
               HeterogeneousMap &options) {
    auto algo = xacc::getAlgorithm(algorithm);
    bool success = algo->initialize(options);
    if (!success) {
      xacc::error("Error initializing " + algorithm + " algorithm.");
    }

    algo->execute(buffer);
  }

  template <typename QuantumKernel, typename... InitialArgs>
  void execute(QuantumKernel &kernel, std::shared_ptr<ObjectiveFunction> objFunction,
               std::shared_ptr<Optimizer> optimizer,
               std::shared_ptr<Observable> observable, InitialArgs... args) {
    auto function = qcor::__internal::getCompositeInstruction(kernel, args...);
    auto nLogicalBits = function->nLogicalBits();
    if (!buffer) {
      buffer = xacc::qalloc(nLogicalBits);
    }

    objFunction->initialize(observable, function, buffer);

    OptFunction optF(
        [&, objFunction](const std::vector<double> &x) { return objFunction->operator()(x); },
        function->nVariables());
    auto result = optimizer->optimize(optF);

    buffer->addExtraInfo("opt-val", ExtraInfo(result.first));
    buffer->addExtraInfo("opt-params", ExtraInfo(result.second));
  }
};

// Full TaskInitiate, built in objective function (given by its name)
template <typename QuantumKernel, typename... InitialArgs>
Handle
taskInitiate(QuantumKernel &&kernel, const std::string objectiveFunctionName,
             std::shared_ptr<Optimizer> optimizer,
             std::shared_ptr<Observable> observable, InitialArgs... args) {
  auto function = qcor::__internal::getCompositeInstruction(kernel, args...);
  auto parameters = __internal::parametersFromVariadic(function, args...);
  return qcor::submit([&, parameters, function](qcor::qpu_handler &q) {
    auto nLogicalBits = function->nLogicalBits();
    auto accelerator = xacc::getAccelerator(function->accelerator_signature());
    auto buffer = xacc::qalloc(nLogicalBits);

    optimizer->appendOption("initial-parameters", parameters);

    HeterogeneousMap m{std::make_pair("observable", observable),
                       std::make_pair("optimizer", optimizer),
                       std::make_pair("accelerator", accelerator),
                       std::make_pair("ansatz", function)};
    q.execute(objectiveFunctionName, kernel, m, parameters);
  });
}

// No observable, assume Z on all qubits
template <typename QuantumKernel, typename... InitialArgs>
Handle taskInitiate(QuantumKernel &&kernel,
                    const std::string objectiveFunctionName,
                    std::shared_ptr<Optimizer> optimizer, InitialArgs... args) {
  return qcor::submit([&](qcor::qpu_handler &q) {
    auto function = qcor::__internal::getCompositeInstruction(kernel, args...);
    auto nLogicalBits = function->nLogicalBits();
    auto accelerator = xacc::getAccelerator(function->accelerator_signature());
    auto buffer = xacc::qalloc(nLogicalBits);
    std::string allZsObsStr = "";
    for (int i = 0; i < nLogicalBits; i++) {
      allZsObsStr += "Z" + std::to_string(i) + " ";
    }
    auto observable = getObservable("pauli", allZsObsStr);
    HeterogeneousMap m{std::make_pair("observable", observable),
                       std::make_pair("optimizer", optimizer),
                       std::make_pair("accelerator", accelerator),
                       std::make_pair("ansatz", function)};
    q.execute(objectiveFunctionName, kernel, m, args...);
  });
}

// No optimizer, evaluate at given initial args
template <typename QuantumKernel, typename... InitialArgs>
Handle
taskInitiate(QuantumKernel &&kernel, const std::string objectiveFunctionName,
             std::shared_ptr<Observable> observable, InitialArgs... args) {
  auto function = qcor::__internal::getCompositeInstruction(kernel, args...);
  auto parameters = __internal::parametersFromVariadic(function, args...);

  return qcor::submit([&, parameters, function](qcor::qpu_handler &q) {
    HeterogeneousMap m{
        std::make_pair("observable", observable),
    };
    q.execute(objectiveFunctionName, kernel, m, parameters);
  });
}

// Custom Objective Function, no Observable
// template <typename QuantumKernel, typename... InitialArgs>
// Handle taskInitiate(QuantumKernel &&kernel, ObjectiveFunction &&objFunction,
//                     std::shared_ptr<Optimizer> optimizer, InitialArgs... args) {
//   return taskInitiate(kernel, objFunction, optimizer, args...);
// }
template <typename QuantumKernel, typename... InitialArgs>
Handle taskInitiate(QuantumKernel &&kernel, std::shared_ptr<ObjectiveFunction> objFunction,
                    std::shared_ptr<Optimizer> optimizer, InitialArgs... args) {
  auto function = qcor::__internal::getCompositeInstruction(kernel, args...);
  auto parameters = __internal::parametersFromVariadic(function, args...);

  return qcor::submit(
      [&, parameters, function, optimizer](qcor::qpu_handler &qh) {

        optimizer->appendOption("initial-parameters", parameters);

        std::string allZsObsStr = "";
        for (int i = 0; i < function->nLogicalBits(); i++) {
          allZsObsStr += "Z" + std::to_string(i) + " ";
        }
        auto observable = getObservable("pauli", allZsObsStr);
        qh.execute(kernel, objFunction, optimizer, observable, args...);
      });
}
} // namespace qcor

#endif
