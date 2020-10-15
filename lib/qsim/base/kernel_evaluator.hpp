#pragma once

#include "objective_function.hpp"
#include "qcor_observable.hpp"
#include "qcor_utils.hpp"
#include "quantum_kernel.hpp"

#include <functional>

namespace qcor {

class KernelFunctor {
protected:
  // Quantum kernel function pointer, we will use
  // this to cast to kernel(composite, args...).
  // Doing it this way means we don't template KernelFunctor
  void *kernel_ptr;

  size_t nbParams;

  // CompositeInstruction representation of the
  // evaluated quantum kernel
  std::shared_ptr<CompositeInstruction> kernel;
  qreg q;

public:
  KernelFunctor(qreg qReg) : q(qReg){};
  qreg &getQreg() { return q; }
  size_t nParams() const { return nbParams; }
  virtual std::shared_ptr<CompositeInstruction>
  evaluate_kernel(const std::vector<double> &in_params) {
    return nullptr;
  }
};

template <typename... KernelArgs>
class KernelFunctorImpl : public KernelFunctor {
private:
  using LocalArgsTranslator = ArgsTranslator<KernelArgs...>;

  std::shared_ptr<CompositeInstruction> create_new_composite() {
    // Create a composite that we can pass to the functor
    std::stringstream name_ss;
    name_ss << this << "_qkernel";
    auto _kernel = qcor::__internal__::create_composite(name_ss.str());
    return _kernel;
  }

protected:
  std::shared_ptr<LocalArgsTranslator> args_translator;
  std::shared_ptr<KernelFunctor> helper;

public:
  KernelFunctorImpl(void *k_ptr,
                    std::shared_ptr<LocalArgsTranslator> translator,
                    std::shared_ptr<KernelFunctor> obj_helper, qreg qReg,
                    size_t nParams)
      : KernelFunctor(qReg) {
    kernel_ptr = k_ptr;
    args_translator = translator;
    helper = obj_helper;
    nbParams = nParams;
  }

  std::shared_ptr<CompositeInstruction>
  evaluate_kernel(const std::vector<double> &in_params) override {
    // Create a new CompositeInstruction, and create a tuple
    // from it so we can concatenate with the tuple args
    auto m_kernel = create_new_composite();
    auto kernel_composite_tuple = std::make_tuple(m_kernel);
    // Translate x parameters into kernel args (represented as a tuple)
    auto translated_tuple = (*args_translator)(in_params);

    // Concatenate the two to make the args list (kernel, args...)
    auto concatenated =
        std::tuple_cat(kernel_composite_tuple, translated_tuple);
    auto kernel_functor = reinterpret_cast<void (*)(
        std::shared_ptr<CompositeInstruction>, KernelArgs...)>(kernel_ptr);
    // Call the functor with those arguments
    qcor::__internal__::evaluate_function_with_tuple_args(kernel_functor,
                                                          concatenated);

    return m_kernel;
  }
};

template <typename... Args>
std::shared_ptr<KernelFunctor> createKernelFunctor(
    void (*quantum_kernel_functor)(std::shared_ptr<CompositeInstruction>,
                                   Args...),
    size_t nQubits, size_t nParams) {

  auto q = qalloc(nQubits);
  auto helper = std::make_shared<KernelFunctor>(q);
  __internal__::ArgsTranslatorAutoGenerator auto_gen;
  auto args_translator = auto_gen(helper->getQreg(), std::tuple<Args...>());

  void *kernel_ptr = reinterpret_cast<void *>(quantum_kernel_functor);

  return std::make_shared<KernelFunctorImpl<Args...>>(
      kernel_ptr, args_translator, helper, q, nParams);
}
} // namespace qcor