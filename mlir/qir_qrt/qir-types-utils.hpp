#pragma once
#include "qir-types.hpp"
#include <typeinfo>
#include <vector>
// Utils for C++ -> QIR type conversion/marshaling
namespace qcor {
namespace qir {
/// Array type
// std::vector -> QIR Array
template <typename T> ::Array *toArray(const std::vector<T> &in_vec) {
  // Integral and Floating-point only
  static_assert(std::is_arithmetic_v<T>,
                "Only support vector of arithmetic types.");
  ::Array *qirArray = new ::Array(in_vec.size(), sizeof(T));
  for (size_t i = 0; i < in_vec.size(); ++i) {
    auto dest = qirArray->getItemPointer(i);
    auto src = &in_vec[i];
    memcpy(dest, src, sizeof(T));
  }
  // TODO: QIR Array supports ref-counting, needs to handle lifetime properly.
  return qirArray;
}

template <typename T> 
std::vector<T> fromArray(::Array *in_array) {
  // Integral and Floating-point only
  static_assert(std::is_arithmetic_v<T>,
                "Only support vector of arithmetic types.");
  if (in_array->element_size() != sizeof(T)) {
    throw std::bad_cast();
  }

  std::vector<T> resultArray;
  for (size_t i = 0; i < in_array->size(); ++i) {
    const T el = *(reinterpret_cast<T *>(in_array->getItemPointer(i)));
    resultArray.emplace_back(el);
  }
  return resultArray;
}

/// Tuple type
// Helper to marshal (pack and unpack) C++ data to Q# QIR data
// e.g. vector is converted to Array type.
template <typename T>
TuplePtr tuple_pack(TuplePtr io_tuple, const std::vector<T> &in_vec) {
  ::Array *qirArray = toArray(in_vec);
  TupleHeader *th = ::TupleHeader::getHeader(io_tuple);
  memcpy(io_tuple, &qirArray, sizeof(::Array *));
  return io_tuple + sizeof(::Array *);
}

// Unpack a value from the tuple and move the pointer to the next element.
template <typename T> TuplePtr tuple_unpack(TuplePtr in_tuple, T &out_val) {
  static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, int64_t>::value,
                "Only support these types now.");
  out_val = *(reinterpret_cast<T *>(in_tuple));
  return in_tuple + sizeof(T);
}

template <typename T>
TuplePtr tuple_unpack(TuplePtr in_tuple, std::vector<T> &out_val) {
  static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, int64_t>::value,
                "Only support vector of these types now.");
  ::Array *arrayPtr = *(reinterpret_cast<::Array **>(in_tuple));
  out_val = fromArray<T>(arrayPtr);
  return in_tuple + sizeof(::Array *);
}

template <typename T> T marshal_one(TuplePtr &io_ptr) {
  T t;
  io_ptr = tuple_unpack(io_ptr, t);
  return t;
}

// Marshal a QIR tuple to std::tuple
template <typename... Args> std::tuple<Args...> marshal(TuplePtr &io_ptr) {
  return std::make_tuple(marshal_one<Args>(io_ptr)...);
}

/// Callable type
// Wrap a std::function in an IFunctor
// which can be used as a Callable.
template <typename ReturnType, typename... Args>
class qs_callback : public qsharp::IFunctor {
public:
  qs_callback(std::function<ReturnType(Args...)> &functor)
      : m_functor(functor) {}
  virtual void execute(TuplePtr args, TuplePtr result) override {
    auto tuple_ptr = args;
    auto args_tuple = marshal<Args...>(tuple_ptr);
    // Pack the result to the return QIR Tuple
    tuple_pack(result, std::apply(m_functor, args_tuple));
  }

private:
  std::function<ReturnType(Args...)> m_functor;
};

// Create a QIR callable from a std::function.
template <typename ReturnType, typename... Args>
Callable *createCallable(std::function<ReturnType(Args...)> &in_func) {
  auto functor = new qs_callback<ReturnType, Args...>(in_func);
  // Create a QIR callable
  return new Callable(functor);
}
} // namespace qir
} // namespace qcor
