#include "qcor.hpp"

// __qpu__ void measure_qbits(qreg q) {
//   for (int i = 0; i < 2; i++) {
//     Measure(q[i]);
//   }
// }

// __qpu__ void quantum_kernel(qreg q, double x) {
//     X(q[0]);
//     Ry(q[1], x);
//     CX(q[1],q[0]);
// }

// __qpu__ void z0z1(qreg q, double x) {
//     quantum_kernel(q, x);
//     measure_qbits(q);
// }

// __qpu__ void check_adjoint(qreg q, double x) {
//     quantum_kernel(q,x);
//     quantum_kernel::adjoint(q,x);
//     measure_qbits(q);
// }

// translated to the following

// the functions will remain, just empty
void measure_qbits(qreg q) {
  void __internal_call_function_measure_qbits(qreg);
  __internal_call_function_measure_qbits(q);
  return;
}

class measure_qbits : public qcor::QuantumKernel<class measure_qbits, qreg> {
  friend class qcor::QuantumKernel<class measure_qbits, qreg>;

protected:
  void operator()(qreg q) {
    if (!parent_kernel) {
      // if has no parent, then create the parent
      // this means this is callable
      parent_kernel = qcor::__internal__::create_composite(kernel_name);
      q.setNameAndStore("q");
    }

    quantum::set_current_program(parent_kernel);

    for (int i = 0; i < 2; i++) {
      quantum::mz(q[i]);
    }
  }

public:
  inline static const std::string kernel_name = "measure_qbits";
  measure_qbits(qreg q) : QuantumKernel<measure_qbits, qreg>(q) {}
  measure_qbits(std::shared_ptr<qcor::CompositeInstruction> p, qreg q)
      : QuantumKernel<measure_qbits, qreg>(p, q) {}

  virtual ~measure_qbits() {
    if (disable_destructor) {
      return;
    }

    quantum::set_backend("qpp", 1024);
    auto [q] = args_tuple;
    operator()(q);
    if (is_callable) {
      quantum::submit(q.results());
    }
  }
};

void measure_qbits(std::shared_ptr<qcor::CompositeInstruction> parent, qreg q) {
  class measure_qbits k(parent, q);
  return;
}

void __internal_call_function_measure_qbits(qreg q) {
  class measure_qbits k(q);
}

void quantum_kernel(qreg q, double x) {
  void __internal_call_function_quantum_kernel(qreg, double);
  __internal_call_function_quantum_kernel(q, x);
  return;
}

class quantum_kernel
    : public qcor::QuantumKernel<class quantum_kernel, qreg, double> {
  friend class qcor::QuantumKernel<class quantum_kernel, qreg, double>;

protected:
  void operator()(qreg q, double t) {
    if (!parent_kernel) {
      // if has no parent, then create the parent
      // this means this is callable
      parent_kernel = qcor::__internal__::create_composite(kernel_name);
      q.setNameAndStore("q");
    }

    quantum::set_current_program(parent_kernel);

    quantum::x(q[0]);
    quantum::ry(q[1], t);
    quantum::cnot(q[1], q[0]);
  }

public:
  inline static const std::string kernel_name = "quantum_kernel";

  quantum_kernel(qreg q, double t)
      : qcor::QuantumKernel<quantum_kernel, qreg, double>(q, t) {}
  quantum_kernel(std::shared_ptr<qcor::CompositeInstruction> p, qreg q,
                 double t)
      : qcor::QuantumKernel<quantum_kernel, qreg, double>(p, q, t) {}
  quantum_kernel() : qcor::QuantumKernel<quantum_kernel, qreg, double>() {}

  virtual ~quantum_kernel() {
    if (disable_destructor) {
      return;
    }
    quantum::set_backend("tnqvm", 1024);
    auto [q, t] = args_tuple;
    operator()(q, t);
    if (is_callable) {
      quantum::submit(q.results());
    }
  }
};

void quantum_kernel(std::shared_ptr<qcor::CompositeInstruction> parent, qreg q,
                    double x) {
  class quantum_kernel k(parent, q, x);
  return;
}

void __internal_call_function_quantum_kernel(qreg q, double x) {
  class quantum_kernel k(q, x);
}

void z0z1(qreg q, double x) {
  void __internal_call_function_z0z1(qreg, double);
  __internal_call_function_z0z1(q, x);
  return;
}

class z0z1 : public qcor::QuantumKernel<class z0z1, qreg, double> {
  friend class qcor::QuantumKernel<class z0z1, qreg, double>;

protected:
  void operator()(qreg q, double t) {
    if (!parent_kernel) {
      // if has no parent, then create the parent
      // this means this is callable
      parent_kernel = qcor::__internal__::create_composite(kernel_name);
      q.setNameAndStore("r");
    }

    quantum::set_current_program(parent_kernel);

    // FIXME Will require qrt_mapper to add parent_kernel to
    // argument list

    quantum_kernel(parent_kernel, q, t);
    measure_qbits(parent_kernel, q);
  }

public:
  inline static const std::string kernel_name = "z0z1";

  z0z1(qreg q, double t) : QuantumKernel<z0z1, qreg, double>(q, t) {}

  virtual ~z0z1() {
    if (disable_destructor) {
      return;
    }
    quantum::set_backend("qpp", 1024);
    auto [q, t] = args_tuple;
    operator()(q, t);
    if (is_callable) {
      quantum::submit(q.results());
    }
  }
};

void __internal_call_function_z0z1(qreg q, double x) { class z0z1 k(q, x); }

void __check_adjoint(qreg q, double x) {
  void __internal_call_function_check_adjoint(qreg, double);
  __internal_call_function_check_adjoint(q, x);
  return;
}

class check_adjoint
    : public qcor::QuantumKernel<class check_adjoint, qreg, double> {
  friend class qcor::QuantumKernel<class check_adjoint, qreg, double>;

protected:
  void operator()(qreg q, double t) {
    if (!parent_kernel) {
      // if has no parent, then create the parent
      // this means this is callable
      parent_kernel = qcor::__internal__::create_composite(kernel_name);
      q.setNameAndStore("v");
    }
    quantum::set_current_program(parent_kernel);

    // FIXMEWill require qrt_mapper to add parent_kernel to
    // argument list, for adjoint too

    quantum_kernel(parent_kernel, q, t);
    quantum_kernel::adjoint(parent_kernel, q, t);
    measure_qbits(parent_kernel, q);
  }

public:
  inline static const std::string kernel_name = "check_adjoint";

  check_adjoint(qreg q, double t)
      : QuantumKernel<check_adjoint, qreg, double>(q, t) {}
  check_adjoint(std::shared_ptr<qcor::CompositeInstruction> p, qreg q, double t)
      : qcor::QuantumKernel<check_adjoint, qreg, double>(p, q, t) {}
  check_adjoint() : qcor::QuantumKernel<check_adjoint, qreg, double>() {}

  virtual ~check_adjoint() {
    if (disable_destructor) {
      return;
    }
    quantum::set_backend("qpp", 1024);
    auto [q, t] = args_tuple;
    operator()(q, t);
    if (is_callable) {
      quantum::submit(q.results());
    }
  }
};

void __internal_call_function_check_adjoint(qreg q, double x) {
  class check_adjoint k(q, x);
}

template <typename... Args>
void test_passing_kernels(void (*quantum_kernel_call)(Args...), Args... args) {
  std::cout << "runnign at the end\n";
  quantum_kernel_call(args...);
}

int main() {
  auto q = qalloc(2);

  quantum_kernel(q, 2.2);

  q.print();

  auto r = qalloc(2);

  z0z1(r, 2.2);
  r.print();

  auto v = qalloc(2);

  check_adjoint(v, 2.2);
  v.print();

  //   check_adjoint::print_kernel(std::cout, v, 2.2);

  auto xx = qalloc(2);
  test_passing_kernels(z0z1, xx, 2.2);
  xx.print();

  auto H = 5.907 - 2.1433 * qcor::X(0) * qcor::X(1) -
           2.1433 * qcor::Y(0) * qcor::Y(1) + .21829 * qcor::Z(0) -
           6.125 * qcor::Z(1);
  auto opt = qcor::createOptimizer("nlopt");
  auto vqe = qcor::createObjectiveFunction("vqe", quantum_kernel, H);
  std::cout << (*vqe)(xx, .59) << "\n";

  // Call taskInitiate, kick off optimization of the give
  // functor dependent on the ObjectiveFunction, async call
  auto handle = qcor::taskInitiate(
      vqe, opt,
      [&](const std::vector<double> x, std::vector<double> &dx) {
       auto e = (*vqe)(xx, x[0]);
        return e;
      },
      1);

  // Go do other work...

  // Query results when ready.
  auto results = qcor::sync(handle);

  // Print the optimal value.
  printf("<H> = %f\n", results.opt_val);
}