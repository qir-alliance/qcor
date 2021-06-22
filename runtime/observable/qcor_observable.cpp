#include "qcor_observable.hpp"

#include "ObservableTransform.hpp"
#include "xacc.hpp"
// #include "xacc_quantum_gate_api.hpp"
// #include "xacc_service.hpp"
#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <cassert>

#include "CompositeInstruction.hpp"
#include "FermionOperator.hpp"
#include "PauliOperator.hpp"
#include "qcor_ir.hpp"
#include "qcor_pimpl_impl.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_quantum_gate_api.hpp"

namespace qcor {

// ---------------- Operator ---------------------- //

class Operator::OperatorImpl
    : public commutative_ring<OperatorImpl>,
      public equality_comparable<OperatorImpl>,
      public commutative_multipliable<OperatorImpl, double>,
      public commutative_multipliable<OperatorImpl, std::complex<double>> {
  friend class Operator;

 private:
  enum operation { PlusEquals, MinusEqual, StarEqual };

  template <typename T>
  struct cast_and_apply {
    void operator()(std::shared_ptr<xacc::Observable> obs, operation _op,
                    const OperatorImpl &other) {
      auto input_casted = std::dynamic_pointer_cast<T>(obs);
      auto other_casted = std::dynamic_pointer_cast<T>(other.op);
      assert(input_casted && other_casted &&
             "Invalid Operator sub-types for arithmetic operation.");

      switch (_op) {
        case operation::PlusEquals:
          input_casted->operator+=(*other_casted.get());
          break;
        case operation::MinusEqual:
          input_casted->operator-=(*other_casted.get());
          break;
        case operation::StarEqual:
          input_casted->operator*=(*other_casted.get());
          break;
        default:
          std::cout << "Invalid Selection\n";
          exit(0);
          break;
      }
    }

    void operator()(std::shared_ptr<xacc::Observable> obs,
                    const double &value) {
      auto input_casted = std::dynamic_pointer_cast<T>(obs);
      assert(input_casted &&
             "Invalid Operator sub-types for arithmetic operation.");
      input_casted->operator*=(value);
    }

    void operator()(std::shared_ptr<xacc::Observable> obs,
                    const std::complex<double> &value) {
      auto input_casted = std::dynamic_pointer_cast<T>(obs);
      assert(input_casted &&
             "Invalid Operator sub-types for arithmetic operation.");
      input_casted->operator*=(value);
    }
  };

  OperatorImpl &apply(const operation &_op, const OperatorImpl &other) {
    if (type == "pauli") {
      cast_and_apply<xacc::quantum::PauliOperator>()(op, _op, other);
    } else if (type == "fermion") {
      cast_and_apply<xacc::quantum::FermionOperator>()(op, _op, other);
    } else {
      // FIXME / TODO As we add more Operator types, update
      // this if/else section
    }
    return *this;
  }

  OperatorImpl &apply(const double &d) {
    if (type == "pauli") {
      cast_and_apply<xacc::quantum::PauliOperator>()(op, d);
    } else if (type == "fermion") {
      cast_and_apply<xacc::quantum::FermionOperator>()(op, d);
    } else {
      // FIXME / TODO As we add more Operator types, update
      // this if/else section
    }
    return *this;
  }

  OperatorImpl &apply(const std::complex<double> &d) {
    if (type == "pauli") {
      cast_and_apply<xacc::quantum::PauliOperator>()(op, d);
    } else if (type == "fermion") {
      cast_and_apply<xacc::quantum::FermionOperator>()(op, d);
    } else {
      // FIXME / TODO As we add more Operator types, update
      // this if/else section
    }
    return *this;
  }

  std::string type;
  std::shared_ptr<xacc::Observable> op;

 public:
  OperatorImpl() = default;
  OperatorImpl(const std::string &_type, const std::string &expr)
      : type(_type) {
    op = xacc::quantum::getObservable(type, expr);
  }
  OperatorImpl(const std::string &_type, std::shared_ptr<xacc::Observable> _op)
      : type(_type), op(_op) {}
  OperatorImpl(const OperatorImpl &other) : type(other.type), op(other.op) {}

  OperatorImpl &operator+=(const OperatorImpl &v) noexcept {
    return apply(operation::PlusEquals, v);
  }

  OperatorImpl &operator-=(const OperatorImpl &v) noexcept {
    return apply(operation::MinusEqual, v);
  }
  OperatorImpl &operator*=(const OperatorImpl &v) noexcept {
    return apply(operation::StarEqual, v);
  }
  OperatorImpl &operator*=(const double v) noexcept { return apply(v); }
  OperatorImpl &operator*=(const std::complex<double> v) noexcept {
    return apply(v);
  }

  bool operator==(const OperatorImpl &v) noexcept {
    if (type == "pauli") {
      auto casted = std::dynamic_pointer_cast<xacc::quantum::PauliOperator>(op);
      auto other_casted =
          std::dynamic_pointer_cast<xacc::quantum::PauliOperator>(v.op);
      assert(casted && other_casted && "Invalid types for Operator == check.");
      return casted->operator==(*other_casted.get());
    } else if (type == "fermion") {
      auto casted =
          std::dynamic_pointer_cast<xacc::quantum::FermionOperator>(op);
      auto other_casted =
          std::dynamic_pointer_cast<xacc::quantum::FermionOperator>(v.op);
      assert(casted && other_casted && "Invalid types for Operator == check.");
      return casted->operator==(*other_casted.get());
    } else {
      // FIXME / TODO As we add more Operator types, update
      // this if/else section
    }
    return false;
  }

  std::vector<Operator> getSubTerms() {
    std::vector<Operator> ret;
    for (auto sub_term : op->getSubTerms()) {
      ret.emplace_back(OperatorImpl(type, sub_term));
    }
    return ret;
  }

  std::vector<Operator> getNonIdentitySubTerms() {
    std::vector<Operator> ret;
    for (auto sub_term : op->getNonIdentitySubTerms()) {
      ret.emplace_back(OperatorImpl(type, sub_term));
    }
    return ret;
  }

  Operator getIdentitySubTerm() {
    auto id_term = op->getIdentitySubTerm();
    if (!id_term) {
      // THROW AN ERROR.
      std::cout << "There is no identity sub term. exiting.\n";
      exit(1);
    }
    return Operator(OperatorImpl(type, op->getIdentitySubTerm()));
  }

  std::complex<double> coefficient() { return op->coefficient(); }

  std::vector<SparseElement> to_sparse_matrix() {
    auto sp_el = op->to_sparse_matrix();
    std::vector<SparseElement> ret;
    for (auto el : sp_el) {
      ret.emplace_back(el.row(), el.col(), el.coeff());
    }
    return ret;
  }

  Operator commutator(Operator &other) {
    return Operator(OperatorImpl(type, op->commutator(other.m_internal->op)));
  }
};

Operator &Operator::operator=(const Operator &other) {
  m_internal->op = other.m_internal->op;
  m_internal->type = other.m_internal->type;
  return *this;
}

Operator::Operator(const std::string &type, const std::string &expr)
    : m_internal(type, expr) {}
Operator::Operator(const Operator &op)
    : m_internal(op.m_internal.operator->()->type,
                 op.m_internal.operator->()->op->toString()) {}
Operator::Operator(const OperatorImpl &&impl)
    : m_internal(impl.type, impl.op) {}

Operator::Operator() = default;
Operator::~Operator() = default;

std::shared_ptr<xacc::Identifiable> Operator::get_as_opaque() {
  return std::dynamic_pointer_cast<xacc::Identifiable>(m_internal->op);
}

Operator &Operator::operator+=(const Operator &v) noexcept {
  OperatorImpl *other = v.m_internal.operator->();
  m_internal->operator+=(*other);
  return *this;
}

Operator &Operator::operator-=(const Operator &v) noexcept {
  OperatorImpl *other = v.m_internal.operator->();
  m_internal->operator-=(*other);
  return *this;
}
Operator &Operator::operator*=(const Operator &v) noexcept {
  OperatorImpl *other = v.m_internal.operator->();
  m_internal->operator*=(*other);
  return *this;
}
bool Operator::operator==(const Operator &v) noexcept {
  OperatorImpl *other = v.m_internal.operator->();
  return m_internal->operator==(*other);
}
Operator &Operator::operator*=(const double v) noexcept {
  m_internal->operator*=(v);
  return *this;
}
Operator &Operator::operator*=(const std::complex<double> v) noexcept {
  m_internal->operator*=(v);
  return *this;
}

int Operator::nQubits() { return m_internal->op->nBits(); }

std::vector<std::shared_ptr<CompositeInstruction>> Operator::observe(
    std::shared_ptr<CompositeInstruction> program) {
  auto opaque = program->get_as_opaque();
  auto as_xacc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(opaque);
  auto cis = m_internal->op->observe(as_xacc);
  std::vector<std::shared_ptr<CompositeInstruction>> ret;
  for (auto ci : cis) {
    ret.emplace_back(std::make_shared<CompositeInstruction>(
        std::dynamic_pointer_cast<xacc::Identifiable>(ci)));
  }

  return ret;
}

// std::vector<CompositeInstruction> Operator::observe(
//     CompositeInstruction &program) {
//   auto cis =
//   m_internal->op->observe(std::dynamic_pointer_cast<xacc::CompositeInstruction>(program->get_as_opaque()));
//   std::vector<CompositeInstruction> ret;
//   for (auto ci : cis) {
//     CompositeInstruction
//     c(std::dynamic_pointer_cast<xacc::Identifiable>(ci));
//     //
//     ret.push_back(CompositeInstruction(std::dynamic_pointer_cast<xacc::Identifiable>(ci)));
//   }

//   return ret;
// }

// std::vector<std::shared_ptr<CompositeInstruction>> Operator::observe(
//     std::shared_ptr<CompositeInstruction> function,
//     const HeterogeneousMap &grouping_options) {}

std::vector<Operator> Operator::getSubTerms() {
  return m_internal->getSubTerms();
}

std::vector<Operator> Operator::getNonIdentitySubTerms() {
  return m_internal->getNonIdentitySubTerms();
}

std::string Operator::toString() const { return m_internal->op->toString(); }
Operator Operator::getIdentitySubTerm() {
  return m_internal->getIdentitySubTerm();
}
std::complex<double> Operator::coefficient() {
  return m_internal->coefficient();
}

std::vector<Operator::SparseElement> Operator::to_sparse_matrix() {
  return m_internal->to_sparse_matrix();
}

Operator Operator::commutator(Operator &op) {
  return m_internal->commutator(op);
}

void __internal_exec_observer(
    xacc::AcceleratorBuffer *b,
    std::vector<std::shared_ptr<CompositeInstruction>> v) {
  // auto vv = v.get()->operator->()->program;
  std::vector<std::shared_ptr<xacc::CompositeInstruction>> tmp;
  std::transform(v.begin(), v.end(), std::back_inserter(tmp),
                 [](std::shared_ptr<CompositeInstruction> c)
                     -> std::shared_ptr<xacc::CompositeInstruction> {
                   return std::dynamic_pointer_cast<xacc::CompositeInstruction>(
                       c->get_as_opaque());
                 });
  xacc::internal_compiler::execute(b, tmp);
}

Operator operator+(double coeff, Operator op) {
  return Operator("pauli", fmt::format("{}", coeff)) + op;
}
Operator operator+(Operator op, double coeff) {
  return op + Operator("pauli", fmt::format("{}", coeff));
}

Operator operator-(double coeff, Operator op) {
  return Operator("pauli", fmt::format("{}", coeff)) - op;
}

Operator operator-(Operator op, double coeff) {
  return op - Operator("pauli", fmt::format("{}", coeff));
}

Operator adag(int idx) { return Operator("fermion", fmt::format("{}^", idx)); }
Operator a(int idx) { return Operator("fermion", fmt::format("{}", idx)); }

Operator X(int idx) { return Operator("pauli", fmt::format("X{}", idx)); }
Operator Y(int idx) { return Operator("pauli", fmt::format("Y{}", idx)); }
Operator Z(int idx) { return Operator("pauli", fmt::format("Z{}", idx)); }

Operator allZs(const int nQubits) {
  auto ret = Z(0);
  for (int i = 1; i < nQubits; i++) {
    ret *= Z(i);
  }
  return ret;
}

Operator SP(int idx) {
  std::complex<double> imag(0.0, 1.0);
  return X(idx) + imag * Y(idx);
}

Operator SM(int idx) {
  std::complex<double> imag(0.0, 1.0);
  return X(idx) - imag * Y(idx);
}

// Eigen::MatrixXcd get_dense_matrix(PauliOperator &op) {
//   auto mat_el = op.to_sparse_matrix();
//   auto size = std::pow(2, op.nBits());
//   Eigen::MatrixXcd mat = Eigen::MatrixXcd::Zero(size, size);
//   for (auto el : mat_el) {
//     mat(el.row(), el.col()) = el.coeff();
//   }
//   return mat;
// }

// Eigen::MatrixXcd get_dense_matrix(std::shared_ptr<Observable> op) {
//   auto mat_el = op->to_sparse_matrix();
//   auto size = std::pow(2, op->nBits());
//   Eigen::MatrixXcd mat = Eigen::MatrixXcd::Zero(size, size);
//   for (auto el : mat_el) {
//     mat(el.row(), el.col()) = el.coeff();
//   }
//   return mat;
// }

Operator createOperator(const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return qcor::Operator("pauli",
                        repr);  // xacc::quantum::getObservable("pauli", repr);
}

Operator createOperator(const std::string &name, const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return qcor::Operator(name,
                        repr);  // xacc::quantum::getObservable(name, repr);
}

Operator createObservable(const std::string &repr) {
  return createOperator(repr);
}

Operator createObservable(const std::string &name, const std::string &repr) {
  return createOperator(name, repr);
}

// std::shared_ptr<Observable> createObservable(const std::string &name,
//                                              HeterogeneousMap &&options) {
//   if (!xacc::isInitialized())
//     xacc::internal_compiler::compiler_InitializeXACC();
//   return xacc::quantum::getObservable(name, options);
// }
// std::shared_ptr<Observable> createObservable(const std::string &name,
//                                              HeterogeneousMap &options) {
//   if (!xacc::isInitialized())
//     xacc::internal_compiler::compiler_InitializeXACC();
//   return xacc::quantum::getObservable(name, options);
// }

// std::shared_ptr<Observable> operatorTransform(const std::string &type,
//                                               qcor::Observable &op) {
//   return xacc::getService<xacc::ObservableTransform>(type)->transform(
//       xacc::as_shared_ptr(&op));
// }
// std::shared_ptr<Observable> operatorTransform(const std::string &type,
//                                               std::shared_ptr<Observable> op)
//                                               {
//   return xacc::getService<xacc::ObservableTransform>(type)->transform(op);
// }

Operator _internal_python_createObservable(const std::string &name,
                                           const std::string &repr) {
  return createOperator(name, repr);
}

namespace __internal__ {
std::map<std::size_t, Operator> cached_observables = {};

std::vector<std::shared_ptr<CompositeInstruction>> observe(
    Operator &obs, std::shared_ptr<CompositeInstruction> program) {
  return obs.observe(program);
}
}  // namespace __internal__

double observe(std::shared_ptr<CompositeInstruction> program, Operator &obs,
               xacc::internal_compiler::qreg &q) {
  // Observe the program
  auto v = obs.observe(program); 

  std::vector<std::shared_ptr<xacc::CompositeInstruction>> tmp;
  std::transform(v.begin(), v.end(), std::back_inserter(tmp),
                 [](std::shared_ptr<CompositeInstruction> c)
                     -> std::shared_ptr<xacc::CompositeInstruction> {
                   return std::dynamic_pointer_cast<xacc::CompositeInstruction>(
                       c->get_as_opaque());
                 });
  xacc::internal_compiler::execute(q.results(), tmp);

  // We want to contract q children buffer
  // exp-val-zs with obs term coeffs
  return q.weighted_sum(
      std::dynamic_pointer_cast<xacc::Observable>(obs.get_as_opaque()).get());
}

// double observe(std::shared_ptr<CompositeInstruction> program, Observable
// &obs,
//                xacc::internal_compiler::qreg &q) {
//   return [program, &obs, &q]() {
//     // Observe the program
//     auto programs = obs.observe(program);

//     xacc::internal_compiler::execute(q.results(), programs);

//     // We want to contract q children buffer
//     // exp-val-zs with obs term coeffs
//     return q.weighted_sum(&obs);
//   }();
// }
}  // namespace qcor

std::ostream &operator<<(std::ostream &os, qcor::Operator const &m) {
  return os << m.toString();
}