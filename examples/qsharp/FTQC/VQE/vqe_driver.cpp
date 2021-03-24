#include <iostream> 
#include <vector>
#include "qir-types.hpp"

// Include the external QSharp function.
qcor_include_qsharp(QCOR__DeuteronVqe__body, double, int64_t shots, Callable* opt_stepper);

// Implement of a callback for Q# via IFunctor interface.
// TODO: this is a rigid impl. for protityping,
// we will handle generic callback signature transformation.
class vqe_callback : public qsharp::IFunctor {
public:
  virtual void execute(TuplePtr args, TuplePtr result) override {
    auto next = unpack(args, m_costVal);
    next = unpack(next, m_previousParams);
    auto _result = internal_execute();
    auto test = pack(result, _result);
  }
  vqe_callback(
      std::function<std::vector<double>(double, std::vector<double>)> functor)
      : m_functor(functor) {}

private:
  std::vector<double> internal_execute() {
    std::vector<double> result = m_functor(m_costVal, m_previousParams);
    return result;
  }

  TuplePtr pack(TuplePtr io_tuple, const std::vector<double> &in_vec) {
    ::Array *qirArray = new ::Array(in_vec.size(), sizeof(double));
    for (size_t i = 0; i < in_vec.size(); ++i) {
      auto dest = qirArray->getItemPointer(i);
      auto src = &in_vec[i];
      memcpy(dest, src, sizeof(double));
    }

    TupleHeader *th = ::TupleHeader::getHeader(io_tuple);
    memcpy(io_tuple, &qirArray, sizeof(::Array *));
    return io_tuple + sizeof(::Array *);
  }

  TuplePtr unpack(TuplePtr in_tuple, double &out_val) {
    out_val = *(reinterpret_cast<double *>(in_tuple));
    return in_tuple + sizeof(double);
  }

  TuplePtr unpack(TuplePtr in_tuple, std::vector<double> &out_val) {
    out_val.clear();
    ::Array *arrayPtr = *(reinterpret_cast<::Array **>(in_tuple));
    // std::cout << "Array of size " << arrayPtr->size()
    //           << "; element size = " << arrayPtr->element_size() << "\n";
    for (size_t i = 0; i < arrayPtr->size(); ++i) {
      const double el =
          *(reinterpret_cast<double *>(arrayPtr->getItemPointer(i)));
      out_val.emplace_back(el);
    }
    return in_tuple + sizeof(::Array *);
  }

private:
  double m_costVal = 0.0;
  std::vector<double> m_previousParams;
  std::function<std::vector<double>(double, std::vector<double>)> m_functor;
};

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qrt ftqc vqe_ansatz.qs vqe_driver.cpp
// Run with:
// $ ./a.out
int main() {
  std::function<std::vector<double>(double, std::vector<double>)> stepper =
      [&](double in_costVal,
          std::vector<double> previous_params) -> std::vector<double> {
    std::cout << "HELLO CALLBACK!\n";
    std::cout << "Cost value = " << in_costVal << "\n";
    return {previous_params[0] + 0.5};
  };

  vqe_callback test(stepper);
  // Create a QIR callable
  Callable cb(&test);

  const double exp_val_xx = QCOR__DeuteronVqe__body(1024, &cb);
  return 0;
}