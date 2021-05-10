#pragma once
#include <optional>
#include "qcor_jit.hpp"
#include "qcor_observable.hpp"
#include "qcor_utils.hpp"
#include "qrt.hpp"

namespace qcor {
enum class QrtType { NISQ, FTQC };

// Forward declare
template <typename... Args> class KernelSignature;

namespace internal {
// KernelSignature is the base of all kernel-like objects
// and we use it to implement kernel modifiers & utilities.
// i.e., anything that is KernelSignature-constructible can use these methods.
template <typename... Args>
void apply_control(std::shared_ptr<CompositeInstruction> parent_kernel,
                   const std::vector<qubit> &ctrl_qbits,
                   KernelSignature<Args...> &kernelCallable, Args... args);

template <typename... Args>
void apply_adjoint(std::shared_ptr<CompositeInstruction> parent_kernel,
                   KernelSignature<Args...> &kernelCallable, Args... args);

template <typename... Args>
double observe(Observable &obs, KernelSignature<Args...> &kernelCallable,
               Args... args);

template <typename... Args>
Eigen::MatrixXcd as_unitary_matrix(KernelSignature<Args...> &kernelCallable,
                                   Args... args);
template <typename... Args>
std::string openqasm(KernelSignature<Args...> &kernelCallable, Args... args);

template <typename... Args>
void print_kernel(KernelSignature<Args...> &kernelCallable, std::ostream &os,
                  Args... args);

template <typename... Args>
std::size_t n_instructions(KernelSignature<Args...> &kernelCallable,
                           Args... args);
} // namespace internal

// The QuantumKernel represents the super-class of all qcor
// quantum kernel functors. Subclasses of this are auto-generated
// via the Clang Syntax Handler capability. Derived classes
// provide a destructor implementation that builds up and
// submits quantum instructions to the specified backend. This enables
// functor-like capability whereby programmers instantiate temporary
// instances of this via the constructor call, and the destructor is
// immediately called. More advanced usage is of course possible for
// qcor developers.
//
// This class works by taking the Derived type (CRTP) and the kernel function
// arguments as template parameters. The Derived type is therefore available for
// instantiation within provided static methods on QuantumKernel. The Args...
// are stored in a member tuple, and are available for use when evaluating the
// kernel. Importantly, QuantumKernel provides static adjoint and ctrl methods
// for auto-generating those circuits.
//
// The Syntax Handler will take kernels like this
// __qpu__ void foo(qreg q) { H(q[0]); }
// and create a derived type of QuantumKernel like this
// class foo : public qcor::QuantumKernel<class foo, qreg> {...};
// with an appropriate implementation of constructors and destructors.
// Users can then call for adjoint/ctrl methods like this
// foo::adjoint(q); foo::ctrl(1, q);
template <typename Derived, typename... Args> class QuantumKernel {
protected:
  // Tuple holder for variadic kernel arguments
  std::tuple<Args...> args_tuple;

  // Parent kernel - null if this is the top-level kernel
  // not null if this is a nested kernel call
  std::shared_ptr<qcor::CompositeInstruction> parent_kernel;

  // Default, submit this kernel, if parent is given
  // turn this to false
  bool is_callable = true;

  // Turn off destructor execution, useful for
  // qcor developers, not to be used by clients / programmers
  bool disable_destructor = false;

public:
  // Flag to indicate we only want to
  // run the pass manager and not execute
  bool optimize_only = false;
  QrtType runtime_env = QrtType::NISQ;
  // Default constructor, takes quantum kernel function arguments
  QuantumKernel(Args... args) : args_tuple(std::forward_as_tuple(args...)) {
    runtime_env = (__qrt_env == "ftqc") ? QrtType::FTQC : QrtType::NISQ;
  }

  // Internal constructor, provide parent kernel, this
  // kernel now represents a nested kernel call and
  // appends to the parent kernel
  QuantumKernel(std::shared_ptr<qcor::CompositeInstruction> _parent_kernel,
                Args... args)
      : args_tuple(std::forward_as_tuple(args...)),
        parent_kernel(_parent_kernel), is_callable(false) {
    runtime_env = (__qrt_env == "ftqc") ? QrtType::FTQC : QrtType::NISQ;
  }

  // Static method for printing this kernel as a flat qasm string
  static void print_kernel(std::ostream &os, Args... args) {
    Derived derived(args...);
    derived.disable_destructor = true;
    derived(args...);
    xacc::internal_compiler::execute_pass_manager();
    os << derived.parent_kernel->toString() << "\n";
  }

  static void print_kernel(Args... args) { print_kernel(std::cout, args...); }

  // Static method to query how many instructions are in this kernel
  static std::size_t n_instructions(Args... args) {
    Derived derived(args...);
    derived.disable_destructor = true;
    derived(args...);
    return derived.parent_kernel->nInstructions();
  }

  // Create the Adjoint of this quantum kernel
  static void adjoint(std::shared_ptr<CompositeInstruction> parent_kernel,
                      Args... args) {
    auto provider = qcor::__internal__::get_provider();

    // instantiate and don't let it call the destructor
    Derived derived(args...);
    derived.disable_destructor = true;

    // run the operator()(args...) call to get the parent_kernel
    derived(args...);

    // get the instructions
    auto instructions = derived.parent_kernel->getInstructions();
    std::shared_ptr<CompositeInstruction> program = derived.parent_kernel;

    // Assert that we don't have measurement
    if (!std::all_of(
            instructions.cbegin(), instructions.cend(),
            [](const auto &inst) { return inst->name() != "Measure"; })) {
      error(
          "Unable to create Adjoint for kernels that have Measure operations.");
    }

    for (int i = 0; i < instructions.size(); i++) {
      auto inst = derived.parent_kernel->getInstruction(i);
      // Parametric gates:
      if (inst->name() == "Rx" || inst->name() == "Ry" ||
          inst->name() == "Rz" || inst->name() == "CPHASE" ||
          inst->name() == "U1" || inst->name() == "CRZ") {
        inst->setParameter(0, -inst->getParameter(0).template as<double>());
      }
      // Handles T and S gates, etc... => T -> Tdg
      else if (inst->name() == "T") {
        auto tdg = provider->createInstruction("Tdg", inst->bits());
        program->replaceInstruction(i, tdg);
      } else if (inst->name() == "S") {
        auto sdg = provider->createInstruction("Sdg", inst->bits());
        program->replaceInstruction(i, sdg);
      }
    }

    // We update/replace instructions in the derived.parent_kernel composite,
    // hence collecting these new instructions and reversing the sequence.
    auto new_instructions = derived.parent_kernel->getInstructions();
    std::reverse(new_instructions.begin(), new_instructions.end());

    // add the instructions to the current parent kernel
    parent_kernel->addInstructions(new_instructions);

    quantum::set_current_program(parent_kernel);

    // no measures, so no execute
  }

  // Create the controlled version of this quantum kernel
  static void ctrl(std::shared_ptr<CompositeInstruction> parent_kernel,
                   const std::vector<int> &ctrlIdx, Args... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrlIdx.size(); i++) {
      ctrl_qubit_vec.push_back({"q", static_cast<size_t>(ctrlIdx[i]), nullptr});
    }
    ctrl(parent_kernel, ctrl_qubit_vec, args...);
  }

  // Single-qubit overload
  static void ctrl(std::shared_ptr<CompositeInstruction> parent_kernel,
                   int ctrlIdx, Args... args) {
    ctrl(parent_kernel, std::vector<int>{ctrlIdx}, args...);
  }

  static void ctrl(std::shared_ptr<CompositeInstruction> parent_kernel,
                   qreg ctrl_qbits, Args... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrl_qbits.size(); i++)
      ctrl_qubit_vec.push_back(ctrl_qbits[i]);

    ctrl(parent_kernel, ctrl_qubit_vec, args...);
  }

  static void ctrl(std::shared_ptr<CompositeInstruction> parent_kernel,
                   const std::vector<qubit> &ctrl_qbits, Args... args) {
    // instantiate and don't let it call the destructor
    Derived derived(args...);
    KernelSignature<Args...> callable(derived);
    internal::apply_control(parent_kernel, ctrl_qbits, callable, args...);
  }

  // Create the controlled version of this quantum kernel
  static void ctrl(std::shared_ptr<CompositeInstruction> parent_kernel,
                   qubit ctrl_qbit, Args... args) {
    ctrl(parent_kernel, std::vector<qubit>{ctrl_qbit}, args...);
  }

  static Eigen::MatrixXcd as_unitary_matrix(Args... args) {
    Derived derived(args...);
    derived.disable_destructor = true;
    derived(args...);
    qcor::KernelToUnitaryVisitor visitor(derived.parent_kernel->nLogicalBits());
    xacc::InstructionIterator iter(derived.parent_kernel);
    while (iter.hasNext()) {
      auto inst = iter.next();
      if (!inst->isComposite() && inst->isEnabled()) {
        inst->accept(&visitor);
      }
    }
    return visitor.getMat();
  }

  static double observe(Observable &obs, Args... args) {
    // instantiate and don't let it call the destructor
    Derived derived(args...);
    derived.disable_destructor = true;

    // run the operator()(args...) call to get the the functor
    // as a CompositeInstruction (derived.parent_kernel)
    derived(args...);

    auto instructions = derived.parent_kernel->getInstructions();
    // Assert that we don't have measurement
    if (!std::all_of(
            instructions.cbegin(), instructions.cend(),
            [](const auto &inst) { return inst->name() != "Measure"; })) {
      error("Unable to observe kernels that already have Measure operations.");
    }

    xacc::internal_compiler::execute_pass_manager();

    // Will fail to compile if more than one qreg is passed.
    std::tuple<Args...> tmp(std::forward_as_tuple(args...));
    auto q = std::get<qreg>(tmp);
    return qcor::observe(derived.parent_kernel, obs, q);
  }

  static double observe(std::shared_ptr<Observable> obs, Args... args) {
    // instantiate and don't let it call the destructor
    Derived derived(args...);
    derived.disable_destructor = true;

    // run the operator()(args...) call to get the the functor
    // as a CompositeInstruction (derived.parent_kernel)
    derived(args...);

    auto instructions = derived.parent_kernel->getInstructions();
    // Assert that we don't have measurement
    if (!std::all_of(
            instructions.cbegin(), instructions.cend(),
            [](const auto &inst) { return inst->name() != "Measure"; })) {
      error("Unable to observe kernels that already have Measure operations.");
    }

    xacc::internal_compiler::execute_pass_manager();

    // Will fail to compile if more than one qreg is passed.
    std::tuple<Args...> tmp(std::forward_as_tuple(args...));
    auto q = std::get<qreg>(tmp);
    return qcor::observe(derived.parent_kernel, obs, q);
  }

  static std::string openqasm(Args... args) {
    Derived derived(args...);
    derived.disable_destructor = true;
    derived(args...);
    xacc::internal_compiler::execute_pass_manager();
    return __internal__::translate("staq", derived.parent_kernel);
  }

  virtual ~QuantumKernel() {}

  template<typename... ArgTypes>
  friend class KernelSignature;
};

// We use the following to enable ctrl operations on our single
// qubit gates, X::ctrl(), Z::ctrl(), H::ctrl(), etc....
template <typename Derived>
using OneQubitKernel = QuantumKernel<Derived, qubit>;

#define ONE_QUBIT_KERNEL_CTRL_ENABLER(CLASSNAME, QRTNAME)                      \
  class CLASSNAME : public OneQubitKernel<class CLASSNAME> {                   \
  public:                                                                      \
    CLASSNAME(qubit q) : OneQubitKernel<CLASSNAME>(q) {}                       \
    CLASSNAME(std::shared_ptr<qcor::CompositeInstruction> _parent_kernel,      \
              qubit q)                                                         \
        : OneQubitKernel<CLASSNAME>(_parent_kernel, q) {                       \
      throw std::runtime_error("you cannot call this.");                       \
    }                                                                          \
    void operator()(qubit q) {                                                 \
      parent_kernel = qcor::__internal__::create_composite(                    \
          "__tmp_one_qubit_ctrl_enabler");                                     \
      quantum::set_current_program(parent_kernel);                             \
      if (runtime_env == QrtType::FTQC) {                                      \
        quantum::set_current_buffer(q.results());                              \
      }                                                                        \
      ::quantum::QRTNAME(q);                                                   \
      return;                                                                  \
    }                                                                          \
    virtual ~CLASSNAME() {}                                                    \
  };

ONE_QUBIT_KERNEL_CTRL_ENABLER(X, x)
ONE_QUBIT_KERNEL_CTRL_ENABLER(Y, y)
ONE_QUBIT_KERNEL_CTRL_ENABLER(Z, z)
ONE_QUBIT_KERNEL_CTRL_ENABLER(H, h)
ONE_QUBIT_KERNEL_CTRL_ENABLER(T, t)
ONE_QUBIT_KERNEL_CTRL_ENABLER(Tdg, tdg)
ONE_QUBIT_KERNEL_CTRL_ENABLER(S, s)
ONE_QUBIT_KERNEL_CTRL_ENABLER(Sdg, sdg)

// The following is a first pass at enabling qcor
// quantum lambdas. The goal is to mimic lambda functionality
// via our QJIT infrastructure. The lambda class takes
// as input a lambda of desired kernel signature calling
// a specific macro which expands to return the function body
// expression as a string, which we use with QJIT jit_compile.
// The lambda class is templated on the types of any capture variables
// the programmer would like to specify, and takes a second constructor
// argument indicating the variable names of all kernel arguments and
// capture variables. Finally, all capture variables must be passed to the
// trailing variadic argument for the lambda class constructor. Once
// instantiated lambda invocation looks just like kernel invocation.

template <typename... CaptureArgs> class _qpu_lambda {
private:
  // Private inner class for getting the type
  // of a capture variable as a string at runtime
  class TupleToTypeArgString {
  protected:
    std::string &tmp;
    std::vector<std::string> var_names;
    int counter = 0;

    template <class T> std::string type_name() {
      typedef typename std::remove_reference<T>::type TR;
      std::unique_ptr<char, void (*)(void *)> own(
          abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
          std::free);
      std::string r = own != nullptr ? own.get() : typeid(TR).name();
      return r;
    }

  public:
    TupleToTypeArgString(std::string &t) : tmp(t) {}
    TupleToTypeArgString(std::string &t, std::vector<std::string> &_var_names)
        : tmp(t), var_names(_var_names) {}
    template <typename T> void operator()(T &t) {
      tmp += type_name<decltype(t)>() + "& " +
             (var_names.empty() ? "arg_" + std::to_string(counter)
                                : var_names[counter]) +
             ",";
      counter++;
    }
  };

  // Kernel lambda source string, has arg structure and body
  std::string &src_str;

  // Capture variable names
  std::string &capture_var_names;

  // By-ref capture variables, stored in tuple
  std::tuple<CaptureArgs &...> capture_vars;

  // Optional capture *by-value* variables:
  // We don't want to make unnecessary copies of capture variables
  // unless explicitly requested ("[=]").
  // Also, some types may not be copy-constructable...
  // Notes:
  // (1) we must copy at the lambda declaration point (i.e. _qpu_lambda
  // constructor)
  // (2) our JIT code chain is constructed using the by-reference convention,
  // just need to handle by-value (copy) at the top-level (i.e., in this tuple
  // storage)
  std::optional<std::tuple<CaptureArgs...>> optional_copy_capture_vars;

  // Quantum Just-in-Time Compiler :)
  QJIT qjit;

public:
  // Constructor, capture vars should be deduced without
  // specifying them since we're using C++17
  _qpu_lambda(std::string &&ff, std::string &&_capture_var_names,
              CaptureArgs &... _capture_vars)
      : src_str(ff), capture_var_names(_capture_var_names),
        capture_vars(std::forward_as_tuple(_capture_vars...)) {
    // Get the original args list
    auto first = src_str.find_first_of("(");
    auto last = src_str.find_first_of(")");
    auto tt = src_str.substr(first, last - first + 1);

    // Get the capture type:
    // By default "[]", pass by reference.
    // [=]: pass by value
    // [&]: pass by reference (same as default)
    const auto first_square_bracket = src_str.find_first_of("[");
    const auto last_square_bracket = src_str.find_first_of("]");
    const auto capture_type = src_str.substr(
        first_square_bracket, last_square_bracket - first_square_bracket + 1);
    if (!capture_type.empty() && capture_type == "[=]") {
      // Store capture vars (by-value)
      optional_copy_capture_vars = std::forward_as_tuple(_capture_vars...);
    }

    // Need to append capture vars to this arg signature
    std::string capture_preamble = "";
    if (!capture_var_names.empty()) {
      std::string args_string = "";
      TupleToTypeArgString co(args_string);
      __internal__::tuple_for_each(capture_vars, co);
      args_string = "," + args_string.substr(0, args_string.length() - 1);
      tt.insert(last - capture_type.size(), args_string);
      capture_preamble += "\n";
      for (auto [i, capture_name] :
           qcor::enumerate(xacc::split(capture_var_names, ','))) {
        // Preamble must use reference type to prevent copy.
        capture_preamble +=
            "auto &" + capture_name + " = arg_" + std::to_string(i) + ";\n";
      }
    }

    // Extract the function body
    first = src_str.find_first_of("{");
    last = src_str.find_last_of("}");
    auto rr = src_str.substr(first, last - first + 1);

    // Reconstruct with new args signature and
    // existing function body
    std::stringstream ss;
    ss << "__qpu__ void foo" << tt << rr;

    // Get as a string, and insert capture
    // preamble if necessary
    auto jit_src = ss.str();
    first = jit_src.find_first_of("{");
    if (!capture_var_names.empty())
      jit_src.insert(first + 1, capture_preamble);

    // std::cout << "JITSRC:\n" << jit_src << "\n";
    // JIT Compile, storing the function pointers
    qjit.jit_compile(jit_src);
  }

  template <typename... FunctionArgs>
  void eval_with_parent(std::shared_ptr<CompositeInstruction> parent,
                        FunctionArgs... args) {
    this->operator()(parent, args...);
  }

  template <typename... FunctionArgs>
  void operator()(std::shared_ptr<CompositeInstruction> parent,
                  FunctionArgs... args) {
    // Map the function args to a tuple
    auto kernel_args_tuple = std::forward_as_tuple(args...);

    if (!optional_copy_capture_vars.has_value()) {
      // By-ref:
      // Merge the function args and the capture vars and execute
      auto final_args_tuple = std::tuple_cat(kernel_args_tuple, capture_vars);
      std::apply(
          [&](auto &&... args) {
            qjit.invoke_with_parent("foo", parent, args...);
          },
          final_args_tuple);
    } else {
      // By-value:
      auto final_args_tuple =
          std::tuple_cat(kernel_args_tuple, optional_copy_capture_vars.value());
      std::apply(
          [&](auto &&... args) {
            qjit.invoke_with_parent("foo", parent, args...);
          },
          final_args_tuple);
    }
  }

  template <typename... FunctionArgs> void operator()(FunctionArgs... args) {
    // Map the function args to a tuple
    auto kernel_args_tuple = std::forward_as_tuple(args...);
    if (!optional_copy_capture_vars.has_value()) {
      // By-ref
      // Merge the function args and the capture vars and execute
      auto final_args_tuple = std::tuple_cat(kernel_args_tuple, capture_vars);
      std::apply([&](auto &&... args) { qjit.invoke("foo", args...); },
                 final_args_tuple);
    } else {
      // By-value
      auto final_args_tuple =
          std::tuple_cat(kernel_args_tuple, optional_copy_capture_vars.value());
      std::apply([&](auto &&... args) { qjit.invoke("foo", args...); },
                 final_args_tuple);
    }
  }

  template<typename... FunctionArgs>
  double observe(std::shared_ptr<Observable> obs, FunctionArgs... args) {
    return observe(*obs.get(), args...);
  }

  template <typename... FunctionArgs>
  double observe(Observable &obs, FunctionArgs... args) {
    auto tempKernel =
        qcor::__internal__::create_composite("temp_lambda_observe");
    this->operator()(tempKernel, args...);

    auto instructions = tempKernel->getInstructions();
    // Assert that we don't have measurement
    if (!std::all_of(
            instructions.cbegin(), instructions.cend(),
            [](const auto &inst) { return inst->name() != "Measure"; })) {
      error("Unable to observe kernels that already have Measure operations.");
    }

    xacc::internal_compiler::execute_pass_manager();

    // Will fail to compile if more than one qreg is passed.
    std::tuple<FunctionArgs...> tmp(std::forward_as_tuple(args...));
    auto q = std::get<qreg>(tmp);
    return qcor::observe(tempKernel, obs, q);
  }

  template <typename... FunctionArgs>
  void ctrl(std::shared_ptr<CompositeInstruction> ir,
            const std::vector<qubit> &ctrl_qbits, FunctionArgs... args) {
    KernelSignature<FunctionArgs...> callable(*this);

    internal::apply_control(ir, ctrl_qbits, callable, args...);
  }

  template <typename... FunctionArgs>
  void ctrl(std::shared_ptr<CompositeInstruction> ir,
            const std::vector<int> &ctrl_idxs, FunctionArgs... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrl_idxs.size(); i++) {
      ctrl_qubit_vec.push_back({"q", static_cast<size_t>(ctrl_idxs[i]), nullptr});
    }
    ctrl(ir, ctrl_qubit_vec, args...);
  }

  template <typename... FunctionArgs>
  void ctrl(std::shared_ptr<CompositeInstruction> ir, int ctrl_qbit,
            FunctionArgs... args) {
    ctrl(ir, std::vector<int>{ctrl_qbit}, args...);
  }

  template <typename... FunctionArgs>
  void ctrl(std::shared_ptr<CompositeInstruction> ir, qubit ctrl_qbit,
            FunctionArgs... args) {
    ctrl(ir, std::vector<qubit>{ctrl_qbit}, args...);
  }

  template <typename... FunctionArgs>
  void ctrl(std::shared_ptr<CompositeInstruction> ir, qreg ctrl_qbits,
            FunctionArgs... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrl_qbits.size(); i++) {
      ctrl_qubit_vec.push_back(ctrl_qbits[i]);
    }
    ctrl(ir, ctrl_qubit_vec, args...);
  }
};

#define qpu_lambda(EXPR, ...) _qpu_lambda(#EXPR, #__VA_ARGS__, ##__VA_ARGS__)

template <typename... Args>
using callable_function_ptr =
    void (*)(std::shared_ptr<xacc::CompositeInstruction>, Args...);

template <typename... Args> class KernelSignature {
private:
  callable_function_ptr<Args...> *readOnly = 0;
  callable_function_ptr<Args...> &function_pointer;
  std::function<void(std::shared_ptr<xacc::CompositeInstruction>, Args...)>
      lambda_func;

public:
  // Here we set function_pointer to null and instead
  // only use lambda_func. If we set lambda_func, function_pointer
  // will never be used, so we should be good.
  template <typename... CaptureArgs>
  KernelSignature(_qpu_lambda<CaptureArgs...> &lambda)
      : function_pointer(*readOnly),
        lambda_func([&](std::shared_ptr<xacc::CompositeInstruction> pp,
                        Args... a) { lambda(pp, a...); }) {}

  KernelSignature(callable_function_ptr<Args...> &&f) : function_pointer(f) {}

  // CTor from a QCOR QuantumKernel instance:
  template <
      typename KernelType,
      std::enable_if_t<
          std::is_base_of_v<QuantumKernel<KernelType, Args...>, KernelType>,
          bool> = true>
  KernelSignature(KernelType &kernel)
      : function_pointer(*readOnly),
        lambda_func(
            [&](std::shared_ptr<xacc::CompositeInstruction> pp, Args... a) {
              // Expand the kernel and append to the *externally-provided*
              // parent kernel as a KernelSignature one.
              kernel.disable_destructor = true;
              kernel(a...);
              pp->addInstructions(kernel.parent_kernel->getInstructions());
            }) {}

  // Ctor from raw void* function pointer.
  // IMPORTANT: since function_pointer is kept as a *reference*,
  // we must keep a reference to the original f_ptr void* as well.
  KernelSignature(void *&f_ptr)
      : function_pointer((callable_function_ptr<Args...> &)f_ptr) {}

  void operator()(std::shared_ptr<xacc::CompositeInstruction> ir,
                  Args... args) {
    if (lambda_func) {
      lambda_func(ir, args...);
      return;
    }

    function_pointer(ir, args...);
  }

  void ctrl(std::shared_ptr<xacc::CompositeInstruction> ir,
            const std::vector<qubit> &ctrl_qbits, Args... args) {
    internal::apply_control(ir, ctrl_qbits, *this, args...);
  }

  void ctrl(std::shared_ptr<xacc::CompositeInstruction> ir,
            const std::vector<int> ctrl_idxs, Args... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrl_idxs.size(); i++) {
      ctrl_qubit_vec.push_back({"q", static_cast<size_t>(ctrl_idxs[i]), nullptr});
    }
    ctrl(ir, ctrl_qubit_vec, args...);
  }
  void ctrl(std::shared_ptr<xacc::CompositeInstruction> ir, int ctrl_qbit,
            Args... args) {
    ctrl(ir, std::vector<int>{ctrl_qbit}, args...);
  }

  void ctrl(std::shared_ptr<xacc::CompositeInstruction> ir, qubit ctrl_qbit,
            Args... args) {
    ctrl(ir, std::vector<qubit>{ctrl_qbit}, args...);
  }

  void ctrl(std::shared_ptr<xacc::CompositeInstruction> ir, qreg ctrl_qbits,
            Args... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrl_qbits.size(); i++) {
      ctrl_qubit_vec.push_back(ctrl_qbits[i]);
    }
    ctrl(ir, ctrl_qubit_vec, args...);
  }

  void adjoint(std::shared_ptr<CompositeInstruction> ir, Args... args) {
    auto tempKernel = qcor::__internal__::create_composite("temp_adjoint");
    operator()(tempKernel, args...);

    // get the instructions
    auto instructions = tempKernel->getInstructions();
    std::shared_ptr<CompositeInstruction> program = tempKernel;

    // Assert that we don't have measurement
    if (!std::all_of(
            instructions.cbegin(), instructions.cend(),
            [](const auto &inst) { return inst->name() != "Measure"; })) {
      error(
          "Unable to create Adjoint for kernels that have Measure operations.");
    }

    auto provider = qcor::__internal__::get_provider();
    for (int i = 0; i < instructions.size(); i++) {
      auto inst = tempKernel->getInstruction(i);
      // Parametric gates:
      if (inst->name() == "Rx" || inst->name() == "Ry" ||
          inst->name() == "Rz" || inst->name() == "CPHASE" ||
          inst->name() == "U1" || inst->name() == "CRZ") {
        inst->setParameter(0, -inst->getParameter(0).template as<double>());
      }
      // Handles T and S gates, etc... => T -> Tdg
      else if (inst->name() == "T") {
        auto tdg = provider->createInstruction("Tdg", inst->bits());
        program->replaceInstruction(i, tdg);
      } else if (inst->name() == "S") {
        auto sdg = provider->createInstruction("Sdg", inst->bits());
        program->replaceInstruction(i, sdg);
      }
    }

    // We update/replace instructions in the derived.parent_kernel composite,
    // hence collecting these new instructions and reversing the sequence.
    auto new_instructions = tempKernel->getInstructions();
    std::reverse(new_instructions.begin(), new_instructions.end());

    // add the instructions to the current parent kernel
    ir->addInstructions(new_instructions);

    ::quantum::set_current_program(ir);
  }
};

namespace internal {
// KernelSignature is the base of all kernel-like objects
// and we use it to implement kernel modifiers && utilities.
// Make this a utility function so that implicit conversion to KernelSignature
// occurs automatically.
template <typename... Args>
void apply_control(std::shared_ptr<CompositeInstruction> parent_kernel,
                   const std::vector<qubit> &ctrl_qbits,
                   KernelSignature<Args...> &kernelCallable, Args... args) {
  const auto buffer_name = ctrl_qbits[0].first;

  for (const auto &qb : ctrl_qbits) {
    if (qb.first != buffer_name) {
      // We can only handle control qubits on the same qReg.
      error("Unable to handle control qubits from different registers");
    }
  }

  std::vector<int> ctrl_bits;
  std::transform(ctrl_qbits.begin(), ctrl_qbits.end(),
                 std::back_inserter(ctrl_bits),
                 [](auto qb) { return qb.second; });

  // Is is in a **compute** segment?
  // i.e. doing control within the compute block itself.
  // need to by-pass the compute marking in order for the control gate to
  // work.
  const bool cached_is_compute_section =
      ::quantum::qrt_impl->isComputeSection();
  if (cached_is_compute_section) {
    ::quantum::qrt_impl->__end_mark_segment_as_compute();
  }

  // Use the controlled gate module of XACC to transform
  auto tempKernel = qcor::__internal__::create_composite("temp_control");
  kernelCallable(tempKernel, args...);

  if (cached_is_compute_section) {
    ::quantum::qrt_impl->__begin_mark_segment_as_compute();
  }

  auto ctrlKernel = qcor::__internal__::create_ctrl_u();
  ctrlKernel->expand({{"U", tempKernel},
                      {"control-idx", ctrl_bits},
                      {"control-buffer", buffer_name}});

  // Mark all the *Controlled* instructions as compute segment
  // if it was in the compute_section.
  // i.e. we have bypassed the marker previously to make C-U to work,
  // now we mark all the generated instructions.
  if (cached_is_compute_section) {
    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      ctrlKernel->getInstruction(instId)->attachMetadata(
          {{"__qcor__compute__segment__", true}});
    }
  }

  for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
    parent_kernel->addInstruction(ctrlKernel->getInstruction(instId));
  }
  // Need to reset and point current program to the parent
  quantum::set_current_program(parent_kernel);
}

template <typename... Args>
void apply_adjoint(std::shared_ptr<CompositeInstruction> parent_kernel,
                   KernelSignature<Args...> &kernelCallable, Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_adjoint");
  kernelCallable(tempKernel, args...);

  // get the instructions
  auto instructions = tempKernel->getInstructions();
  std::shared_ptr<CompositeInstruction> program = tempKernel;

  // Assert that we don't have measurement
  if (!std::all_of(
          instructions.cbegin(), instructions.cend(),
          [](const auto &inst) { return inst->name() != "Measure"; })) {
    error("Unable to create Adjoint for kernels that have Measure operations.");
  }

  auto provider = qcor::__internal__::get_provider();
  for (int i = 0; i < instructions.size(); i++) {
    auto inst = tempKernel->getInstruction(i);
    // Parametric gates:
    if (inst->name() == "Rx" || inst->name() == "Ry" || inst->name() == "Rz" ||
        inst->name() == "CPHASE" || inst->name() == "U1" ||
        inst->name() == "CRZ") {
      inst->setParameter(0, -inst->getParameter(0).template as<double>());
    }
    // Handles T and S gates, etc... => T -> Tdg
    else if (inst->name() == "T") {
      auto tdg = provider->createInstruction("Tdg", inst->bits());
      program->replaceInstruction(i, tdg);
    } else if (inst->name() == "S") {
      auto sdg = provider->createInstruction("Sdg", inst->bits());
      program->replaceInstruction(i, sdg);
    }
  }

  // We update/replace instructions in the derived.parent_kernel composite,
  // hence collecting these new instructions and reversing the sequence.
  auto new_instructions = tempKernel->getInstructions();
  std::reverse(new_instructions.begin(), new_instructions.end());

  // Are we in a compute section?
  // Make sure we mark all instructions appropriately.
  // e.g. we do create new instructions, e.g. Tdg, Sdg, in the above step.
  if (::quantum::qrt_impl->isComputeSection()) {
    for (auto &inst : new_instructions) {
      inst->attachMetadata({{"__qcor__compute__segment__", true}});
    }
  }

  // add the instructions to the current parent kernel
  parent_kernel->addInstructions(new_instructions);

  ::quantum::set_current_program(parent_kernel);
}

template <typename... Args>
double observe(Observable &obs, KernelSignature<Args...> &kernelCallable,
                    Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_observe");
  kernelCallable(tempKernel, args...);
  auto instructions = tempKernel->getInstructions();
  // Assert that we don't have measurement
  if (!std::all_of(
          instructions.cbegin(), instructions.cend(),
          [](const auto &inst) { return inst->name() != "Measure"; })) {
    error("Unable to observe kernels that already have Measure operations.");
  }

  xacc::internal_compiler::execute_pass_manager();

  // Will fail to compile if more than one qreg is passed.
  std::tuple<Args...> tmp(std::forward_as_tuple(args...));
  auto q = std::get<qreg>(tmp);
  return qcor::observe(tempKernel, obs, q);
}

template <typename... Args>
Eigen::MatrixXcd as_unitary_matrix(KernelSignature<Args...> &kernelCallable,
                                   Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_as_unitary");
  kernelCallable(tempKernel, args...);
  auto instructions = tempKernel->getInstructions();
  // Assert that we don't have measurement
  if (!std::all_of(
          instructions.cbegin(), instructions.cend(),
          [](const auto &inst) { return inst->name() != "Measure"; })) {
    error("Unable to compute unitary matrix for kernels that already have "
          "Measure operations.");
  }
  qcor::KernelToUnitaryVisitor visitor(tempKernel->nLogicalBits());
  xacc::InstructionIterator iter(tempKernel);
  while (iter.hasNext()) {
    auto inst = iter.next();
    if (!inst->isComposite() && inst->isEnabled()) {
      inst->accept(&visitor);
    }
  }
  return visitor.getMat();
}

template <typename... Args>
std::string openqasm(KernelSignature<Args...> &kernelCallable, Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_as_openqasm");
  kernelCallable(tempKernel, args...);
  xacc::internal_compiler::execute_pass_manager();
  return __internal__::translate("staq", tempKernel);
}

template <typename... Args>
void print_kernel(KernelSignature<Args...> &kernelCallable, std::ostream &os,
             Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_print");
  kernelCallable(tempKernel, args...);
  xacc::internal_compiler::execute_pass_manager();
  os << tempKernel->toString() << "\n";
}

template <typename... Args>
std::size_t n_instructions(KernelSignature<Args...> &kernelCallable,
                           Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_count_insts");
  kernelCallable(tempKernel, args...);
  return tempKernel->nInstructions();
}
} // namespace internal
} // namespace qcor
