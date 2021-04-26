#pragma once
// Helper macros to generate QCOR kernel wrapper for
// external Q# kernel (compiled to QIR)
#include "qir-types.hpp"

#define _STRINGIZE(arg) _STRINGIZE1(arg)
#define _STRINGIZE1(arg) _STRINGIZE2(arg)
#define _STRINGIZE2(arg) #arg

#define CONCATENATE(arg1, arg2) CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2) CONCATENATE2(arg1, arg2)
#define CONCATENATE2(arg1, arg2) arg1##arg2

#define FOR_EACH_1(what, x, ...) what(x, 1)
#define FOR_EACH_2(what, x, ...) what(x, 2) FOR_EACH_1(what, __VA_ARGS__)
#define FOR_EACH_3(what, x, ...) what(x, 3) FOR_EACH_2(what, __VA_ARGS__)
#define FOR_EACH_4(what, x, ...) what(x, 4) FOR_EACH_3(what, __VA_ARGS__)
#define FOR_EACH_5(what, x, ...) what(x, 5) FOR_EACH_4(what, __VA_ARGS__)
#define FOR_EACH_6(what, x, ...) what(x, 6) FOR_EACH_5(what, __VA_ARGS__)
#define FOR_EACH_7(what, x, ...) what(x, 7) FOR_EACH_6(what, __VA_ARGS__)
#define FOR_EACH_8(what, x, ...) what(x, 8) FOR_EACH_7(what, __VA_ARGS__)

#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define FOR_EACH_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define FOR_EACH_(N, what, ...) CONCATENATE(FOR_EACH_, N)(what, __VA_ARGS__)
#define FOR_EACH(what, ...)                                                    \
  FOR_EACH_(FOR_EACH_NARG(__VA_ARGS__), what, __VA_ARGS__)

#define CONSTRUCT_ARGS_LIST_(type_name, var_name) , type_name var_name
#define CONSTRUCT_ARGS_LIST(type_name, counter)                                \
  CONSTRUCT_ARGS_LIST_(type_name, __internal__var__##counter)
#define CONSTRUCT_VAR_NAME_LIST(type_name, counter) , __internal__var__##counter

// Macro to use to construct list of types and var names.
#define ARGS_LIST_FOR_FUNC_SIGNATURE(...)                                      \
  FOR_EACH(CONSTRUCT_ARGS_LIST, __VA_ARGS__)
#define ARGS_LIST_FOR_FUNC_INVOKE(...)                                         \
  FOR_EACH(CONSTRUCT_VAR_NAME_LIST, __VA_ARGS__)

#define GET_MACRO(_0, _1, _2, NAME, ...) NAME
#define qcor_import_qsharp_kernel(...)                                         \
  GET_MACRO(_0, ##__VA_ARGS__, qcor_import_qsharp_kernel_var,                  \
            qcor_import_qsharp_kernel_no_var, unknown)                         \
  (__VA_ARGS__)

#define qcor_import_qsharp_kernel_var(OPERATION_NAME, ...)                     \
  extern "C" void OPERATION_NAME##__body(::Array *, __VA_ARGS__);              \
  class OPERATION_NAME                                                         \
      : public qcor::QuantumKernel<class OPERATION_NAME, qreg, __VA_ARGS__> {  \
  private:                                                                     \
    ::Array *m_qubits = nullptr;                                               \
    friend class qcor::QuantumKernel<class OPERATION_NAME, qreg, __VA_ARGS__>; \
                                                                               \
  protected:                                                                   \
    void operator()(qreg q ARGS_LIST_FOR_FUNC_SIGNATURE(__VA_ARGS__)) {        \
      if (!parent_kernel) {                                                    \
        parent_kernel = qcor::__internal__::create_composite(kernel_name);     \
      }                                                                        \
      quantum::set_current_program(parent_kernel);                             \
      if (runtime_env == QrtType::FTQC) {                                      \
        quantum::set_current_buffer(q.results());                              \
      }                                                                        \
      /* Convert the qreg to QIR Array of Qubits */                            \
      if (!m_qubits) {                                                         \
        m_qubits = new ::Array(q.size());                                      \
        for (int i = 0; i < q.size(); ++i) {                                   \
          auto qubit = Qubit::allocate();                                      \
          int8_t *arrayPtr = (*m_qubits)[i];                                   \
          auto qubitPtr = reinterpret_cast<Qubit **>(arrayPtr);                \
          *qubitPtr = qubit;                                                   \
        }                                                                      \
      }                                                                        \
      OPERATION_NAME##__body(m_qubits ARGS_LIST_FOR_FUNC_INVOKE(               \
          __VA_ARGS__)); /* std::cout << "INVOKE:\n" <<                        \
                            parent_kernel->toString(); */                      \
    }                                                                          \
                                                                               \
  public:                                                                      \
    inline static const std::string kernel_name = #OPERATION_NAME;             \
    OPERATION_NAME(qreg q ARGS_LIST_FOR_FUNC_SIGNATURE(__VA_ARGS__))           \
        : QuantumKernel<OPERATION_NAME, qreg, __VA_ARGS__>(                    \
              q ARGS_LIST_FOR_FUNC_INVOKE(__VA_ARGS__)) {}                     \
    OPERATION_NAME(std::shared_ptr<qcor::CompositeInstruction> _parent,        \
                   qreg q ARGS_LIST_FOR_FUNC_SIGNATURE(__VA_ARGS__))           \
        : QuantumKernel<OPERATION_NAME, qreg, __VA_ARGS__>(                    \
              _parent, q ARGS_LIST_FOR_FUNC_INVOKE(__VA_ARGS__)) {}            \
    virtual ~OPERATION_NAME() {                                                \
      if (disable_destructor) {                                                \
        return;                                                                \
      }                                                                        \
      auto [q ARGS_LIST_FOR_FUNC_INVOKE(__VA_ARGS__)] = args_tuple;            \
      operator()(q ARGS_LIST_FOR_FUNC_INVOKE(__VA_ARGS__));                    \
      xacc::internal_compiler::execute_pass_manager();                         \
      if (is_callable) {                                                       \
        quantum::submit(q.results());                                          \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  void OPERATION_NAME(qreg q ARGS_LIST_FOR_FUNC_SIGNATURE(__VA_ARGS__)) {      \
    class OPERATION_NAME kernel(q ARGS_LIST_FOR_FUNC_INVOKE(__VA_ARGS__));     \
  }

#define qcor_import_qsharp_kernel_no_var(OPERATION_NAME)                       \
  extern "C" void OPERATION_NAME##__body(::Array *);                           \
  class OPERATION_NAME                                                         \
      : public qcor::QuantumKernel<class OPERATION_NAME, qreg> {               \
  private:                                                                     \
    ::Array *m_qubits = nullptr;                                               \
    friend class qcor::QuantumKernel<class OPERATION_NAME, qreg>;              \
                                                                               \
  protected:                                                                   \
    void operator()(qreg q) {                                                  \
      if (!parent_kernel) {                                                    \
        std::cout << "Create parent kernel\n"; \
        parent_kernel = qcor::__internal__::create_composite(kernel_name);     \
      }                                                                        \
      std::cout << "Parent:\n" << parent_kernel->toString() << "\n";           \
      quantum::set_current_program(parent_kernel);                             \
      if (runtime_env == QrtType::FTQC) {                                      \
        quantum::set_current_buffer(q.results());                              \
      }                                                                        \
      /* Convert the qreg to QIR Array of Qubits */                            \
      if (!m_qubits) {                                                         \
        m_qubits = new ::Array(q.size());                                      \
        for (int i = 0; i < q.size(); ++i) {                                   \
          auto qubit = Qubit::allocate();                                      \
          int8_t *arrayPtr = (*m_qubits)[i];                                   \
          auto qubitPtr = reinterpret_cast<Qubit **>(arrayPtr);                \
          *qubitPtr = qubit;                                                   \
        }                                                                      \
      }                                                                        \
      std::cout << "INVOKE:\n" << parent_kernel->toString() << "\n";           \
      OPERATION_NAME##__body(m_qubits); /* std::cout << "INVOKE:\n" <<         \
                            parent_kernel->toString(); */                      \
    }                                                                          \
                                                                               \
  public:                                                                      \
    inline static const std::string kernel_name = #OPERATION_NAME;             \
    OPERATION_NAME(qreg q) : QuantumKernel<OPERATION_NAME, qreg>(q) {}         \
    OPERATION_NAME(std::shared_ptr<qcor::CompositeInstruction> _parent,        \
                   qreg q)                                                     \
        : QuantumKernel<OPERATION_NAME, qreg>(_parent, q) {}                   \
    virtual ~OPERATION_NAME() {                                                \
      if (disable_destructor) {                                                \
        return;                                                                \
      }                                                                        \
      auto [q] = args_tuple;                                                   \
      std::cout << "here\n"; \
      operator()(q);                                                           \
      xacc::internal_compiler::execute_pass_manager();                         \
      if (is_callable) {                                                       \
        quantum::submit(q.results());                                          \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  void OPERATION_NAME(std::shared_ptr<qcor::CompositeInstruction> parent,      \
                      qreg q) {                                                \
    class OPERATION_NAME kernel(parent, q);                                    \
  }                                                                            \
  void OPERATION_NAME(qreg q) { class OPERATION_NAME kernel(q); }              

// Usage:
// qcor_import_qsharp_kernel(MyQsharpKernel, double, int);
// This will inject a QCOR kernel of signature:
// void MyQsharpKernel(qreg, double, int);
// which can be used in QCOR, e.g. supporting NISQ remote submit API
// and runtime pass manager, etc.
