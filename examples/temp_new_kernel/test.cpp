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
void measure_bits(qreg q) {
    return;
}

void quantum_kernel(qreg q, double x) {
    return;
}

void z0z1(qreg q, double x) {
    return;
}

void check_adjoint(qreg q, double x) {
    return;
}

class measure_qbits : public qcor::QuantumKernel<measure_qbits, qreg> {
  friend class qcor::QuantumKernel<measure_qbits, qreg>;

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

class quantum_kernel
    : public qcor::QuantumKernel<quantum_kernel, qreg, double> {
  friend class qcor::QuantumKernel<quantum_kernel, qreg, double>;

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
      : QuantumKernel<quantum_kernel, qreg, double>(q, t) {}
  quantum_kernel(std::shared_ptr<qcor::CompositeInstruction> p, qreg q,
                 double t)
      : QuantumKernel<quantum_kernel, qreg, double>(p, q, t) {}
  quantum_kernel() : QuantumKernel<quantum_kernel, qreg, double>() {}

  virtual ~quantum_kernel() {
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

class z0z1 : public qcor::QuantumKernel<z0z1, qreg, double> {

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

class check_adjoint : public qcor::QuantumKernel<check_adjoint, qreg, double> {

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
    std::cout << "check here\n" << parent_kernel->toString() << "\n";
    measure_qbits(parent_kernel, q);
  }

public:
  inline static const std::string kernel_name = "check_adjoint";

  check_adjoint(qreg q, double t)
      : QuantumKernel<check_adjoint, qreg, double>(q, t) {}

  virtual ~check_adjoint() {
    if (disable_destructor) {
      return;
    }
    quantum::set_backend("qpp", 1024);
    auto [q, t] = args_tuple;
    operator()(q, t);
    if (is_callable) {
      //   quantum::program = parent_kernel;
      std::cout << quantum::program->toString() << "\n";
      quantum::submit(q.results());
    }
  }
};

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
}