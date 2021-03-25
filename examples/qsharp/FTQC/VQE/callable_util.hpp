#pragma once
#include "qir-types.hpp"

namespace qcor {
namespace qsharp {

// Helper to marshal (pack and unpack) C++ data to Q# QIR data
// e.g. vector is converted to Array type.
template <typename T>
TuplePtr pack(TuplePtr io_tuple, const std::vector<T> &in_vec) {
  static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, int64_t>::value,
                "Only support vector of these types now.");
  ::Array *qirArray = new ::Array(in_vec.size(), sizeof(T));
  for (size_t i = 0; i < in_vec.size(); ++i) {
    auto dest = qirArray->getItemPointer(i);
    auto src = &in_vec[i];
    memcpy(dest, src, sizeof(T));
  }

  TupleHeader *th = ::TupleHeader::getHeader(io_tuple);
  memcpy(io_tuple, &qirArray, sizeof(::Array *));
  return io_tuple + sizeof(::Array *);
}

// Unpack a value from the tuple and move the pointer to the next element.
template <typename T> TuplePtr unpack(TuplePtr in_tuple, T &out_val) {
  static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, int64_t>::value,
                "Only support these types now.");
  out_val = *(reinterpret_cast<T *>(in_tuple));
  return in_tuple + sizeof(T);
}

template <typename T>
TuplePtr unpack(TuplePtr in_tuple, std::vector<T> &out_val) {
  static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, int64_t>::value,
                "Only support vector of these types now.");
  out_val.clear();
  ::Array *arrayPtr = *(reinterpret_cast<::Array **>(in_tuple));
  for (size_t i = 0; i < arrayPtr->size(); ++i) {
    const double el = *(reinterpret_cast<T *>(arrayPtr->getItemPointer(i)));
    out_val.emplace_back(el);
  }
  return in_tuple + sizeof(::Array *);
}

template <typename ReturnType, typename... Args>
class qs_callback : public qsharp::IFunctor {
public:
  virtual void execute(TuplePtr args, TuplePtr result) override {
    auto next = qsharp::unpack(args, m_costVal);
    next = qsharp::unpack(next, m_previousParams);
    auto _result = internal_execute();
    auto test = qsharp::pack(result, _result);
  }
  qs_callback(std::function<ReturnType(Args...)> &functor)
      : m_functor(functor) {}

private:
  std::vector<double> internal_execute() {
    std::vector<double> result = m_functor(m_costVal, m_previousParams);
    return result;
  }

private:
  double m_costVal = 0.0;
  std::vector<double> m_previousParams;
  std::function<ReturnType(Args...)> m_functor;
};

template <typename ReturnType, typename... Args>
Callable *createCallable(std::function<ReturnType(Args...)> &in_func) {
  auto functor = new qs_callback<ReturnType, Args...>(in_func);
  // Create a QIR callable
  return new Callable(functor);
}
} // namespace qsharp
} // namespace qcor