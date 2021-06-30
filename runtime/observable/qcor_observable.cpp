#include "qcor_observable.hpp"

#include "ObservableTransform.hpp"
#include "xacc.hpp"
// #include "xacc_quantum_gate_api.hpp"
#include <spdlog/fmt/fmt.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cassert>

#include "CompositeInstruction.hpp"
#include "FermionOperator.hpp"
#include "PauliOperator.hpp"
#include "qalloc.hpp"
#include "qcor_ir.hpp"
#include "qcor_pimpl_impl.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_quantum_gate_api.hpp"
#include "xacc_service.hpp"

namespace qcor {

// ---------------- Operator ---------------------- //

// Internal hidden implementation
class Operator::OperatorImpl
    : public commutative_ring<OperatorImpl>,
      public equality_comparable<OperatorImpl>,
      public commutative_multipliable<OperatorImpl, double>,
      public commutative_multipliable<OperatorImpl, std::complex<double>> {
  friend class Operator;

 private:
  enum operation { PlusEquals, MinusEqual, StarEqual };

  // This function performs sub-class specific algebraic operation 
  // with another operator instance, or scalar value (double, complex)
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

  // Apply the desired operation between this operator an the given other operator
  OperatorImpl &apply(const operation &_op, const OperatorImpl &other) {
    if (type == "pauli") {
      cast_and_apply<xacc::quantum::PauliOperator>()(op, _op, other);
    } else if (type == "fermion") {
      cast_and_apply<xacc::quantum::FermionOperator>()(op, _op, other);
    } else {
      // FIXME / TODO As we add more Operator types, update
      // this if/else section
      error("Invalid Operator type for the given algebraic operation.");
    }
    return *this;
  }

  // Apply the desired operation between this operator an the given scalar double
  OperatorImpl &apply(const double &d) {
    if (type == "pauli") {
      cast_and_apply<xacc::quantum::PauliOperator>()(op, d);
    } else if (type == "fermion") {
      cast_and_apply<xacc::quantum::FermionOperator>()(op, d);
    } else {
      // FIXME / TODO As we add more Operator types, update
      // this if/else section
      error("Invalid Operator type for the given algebraic operation.");
    }
    return *this;
  }

  // Apply the desired operation between this operator an the given scalar complex
  OperatorImpl &apply(const std::complex<double> &d) {
    if (type == "pauli") {
      cast_and_apply<xacc::quantum::PauliOperator>()(op, d);
    } else if (type == "fermion") {
      cast_and_apply<xacc::quantum::FermionOperator>()(op, d);
    } else {
      // FIXME / TODO As we add more Operator types, update
      // this if/else section
      error("Invalid Operator type for the given algebraic operation.");
    }
    return *this;
  }

  std::string type;
  std::shared_ptr<xacc::Observable> op;

 public:
  // Internal impl constructed from HetMap of options, or string-like expression
  OperatorImpl() = default;
  OperatorImpl(const std::string &_type, const std::string &expr)
      : type(_type) {
    op = xacc::quantum::getObservable(type, expr);
  }

  OperatorImpl(const std::string &name, xacc::HeterogeneousMap &options) {
    auto tmp_op = xacc::quantum::getObservable(name, options);
    auto obs_str = tmp_op->toString();

    if (obs_str.find("^") != std::string::npos) {

      op = xacc::quantum::getObservable("fermion", obs_str);
      type = "fermion";

    } else if (obs_str.find("X") != std::string::npos ||
               obs_str.find("Y") != std::string::npos ||
               obs_str.find("Z") != std::string::npos) {
      op = xacc::quantum::getObservable("pauli", obs_str);
      type = "pauli";
    }
  }

  OperatorImpl(const std::string &_type, std::shared_ptr<xacc::Observable> _op)
      : type(_type), op(_op) {}
  OperatorImpl(const OperatorImpl &other) : type(other.type), op(other.op) {}

  // Implement internal Algebraic API
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
     error("There is no identity sub term. exiting.");
    }
    return Operator(OperatorImpl(type, op->getIdentitySubTerm()));
  }

  bool hasIdentitySubTerm() { return op->getIdentitySubTerm() != nullptr; }
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

  std::pair<std::vector<int>, std::vector<int>> toBinaryVectors(
      const int nQubits) {
    assert(type == "pauli" && "toBinaryVectors only works for pauli operators");
    return std::dynamic_pointer_cast<xacc::quantum::PauliOperator>(op)
        ->toBinaryVectors(nQubits);
  }

  void mapQubitSites(std::map<int, int> &siteMap) {
    assert(type == "pauli" && "mapQubitSites only works for pauli operators");
    std::dynamic_pointer_cast<xacc::quantum::PauliOperator>(op)->mapQubitSites(
        siteMap);
  }
};

Operator &Operator::operator=(const Operator &other) {
  m_internal->op = other.m_internal->op;
  m_internal->type = other.m_internal->type;
  return *this;
}

Operator::Operator(const std::string &name, xacc::HeterogeneousMap &options)
    : m_internal(name, options) {}
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

std::pair<std::vector<int>, std::vector<int>> Operator::toBinaryVectors(
    const int nQubits) {
  return m_internal->toBinaryVectors(nQubits);
}
void Operator::mapQubitSites(std::map<int, int> &siteMap) {
  return m_internal->mapQubitSites(siteMap);
}

std::vector<std::shared_ptr<CompositeInstruction>> Operator::observe(
    std::shared_ptr<CompositeInstruction> program) {
  auto as_xacc = program->as_xacc();
  auto cis = m_internal->op->observe(as_xacc);
  std::vector<std::shared_ptr<CompositeInstruction>> ret;
  for (auto ci : cis) {
    ret.emplace_back(std::make_shared<CompositeInstruction>(
        std::dynamic_pointer_cast<xacc::Identifiable>(ci)));
  }

  return ret;
}

std::vector<Operator> Operator::getSubTerms() {
  return m_internal->getSubTerms();
}

std::vector<Operator> Operator::getNonIdentitySubTerms() {
  return m_internal->getNonIdentitySubTerms();
}
bool Operator::hasIdentitySubTerm() { return m_internal->hasIdentitySubTerm(); }

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

Operator Operator::transform(const std::string &type,
                             xacc::HeterogeneousMap m) {
  auto transformed =
      xacc::getService<xacc::ObservableTransform>(type)->transform(
          m_internal->op);
  auto new_type =
      std::dynamic_pointer_cast<xacc::quantum::FermionOperator>(transformed)
          ? "fermion"
          : "pauli";
  return Operator(OperatorImpl(new_type, transformed));
}

void __internal_exec_observer(
    xacc::AcceleratorBuffer *b,
    std::vector<std::shared_ptr<CompositeInstruction>> v) {
  // auto vv = v.get()->operator->()->program;
  std::vector<std::shared_ptr<xacc::CompositeInstruction>> tmp;
  std::transform(v.begin(), v.end(), std::back_inserter(tmp),
                 [](std::shared_ptr<CompositeInstruction> c)
                     -> std::shared_ptr<xacc::CompositeInstruction> {
                   return c->as_xacc();
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

Operator adag(int idx) {
  return Operator("fermion", fmt::format("1.0 {}^", idx));
}
Operator a(int idx) { return Operator("fermion", fmt::format("1.0 {}", idx)); }

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

Eigen::MatrixXcd get_dense_matrix(Operator &op) {
  auto mat_el = op.to_sparse_matrix();
  auto size = std::pow(2, op.nBits());
  Eigen::MatrixXcd mat = Eigen::MatrixXcd::Zero(size, size);
  for (auto el : mat_el) {
    mat(el.row(), el.col()) = el.coeff();
  }
  return mat;
}

Operator createOperator(const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return qcor::Operator("pauli", repr);
}

Operator createOperator(const std::string &name, const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return qcor::Operator(name, repr);
}

Operator createOperator(const std::string &name, HeterogeneousMap &&options) {
  return createOperator(name, options);
}
Operator createOperator(const std::string &name, HeterogeneousMap &options) {
  return qcor::Operator(name, options);
}

Operator createObservable(const std::string &repr) {
  return createOperator(repr);
}

Operator createObservable(const std::string &name, const std::string &repr) {
  return createOperator(name, repr);
}

Operator operatorTransform(const std::string &type, Operator &op) {
  return op.transform(type);
}

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
                   return c->as_xacc();
                 });
  xacc::internal_compiler::execute(q.results(), tmp);

  // We want to contract q children buffer
  // exp-val-zs with obs term coeffs
  return q.weighted_sum(
      std::dynamic_pointer_cast<xacc::Observable>(obs.get_as_opaque()).get());
}

}  // namespace qcor

std::ostream &operator<<(std::ostream &os, qcor::Operator const &m) {
  return os << m.toString();
}