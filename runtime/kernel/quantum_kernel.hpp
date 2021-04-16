#pragma once

#include "qcor_jit.hpp"
#include "qcor_observable.hpp"
#include "qcor_utils.hpp"
#include "qrt.hpp"

namespace qcor {
enum class QrtType { NISQ, FTQC };

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
template <typename Derived, typename... Args>
class QuantumKernel {
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
        parent_kernel(_parent_kernel),
        is_callable(false) {
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

  static void print_kernel(Args... args) {
    Derived derived(args...);
    derived.disable_destructor = true;
    derived(args...);
    xacc::internal_compiler::execute_pass_manager();
    std::cout << derived.parent_kernel->toString() << "\n";
  }

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
    // instantiate and don't let it call the destructor
    Derived derived(args...);
    derived.disable_destructor = true;

    // run the operator()(args...) call to get the the functor
    // as a CompositeInstruction (derived.parent_kernel)
    derived(args...);

    // Use the controlled gate module of XACC to transform
    auto tempKernel = qcor::__internal__::create_composite("temp_control");
    tempKernel->addInstruction(derived.parent_kernel);

    auto ctrlKernel = qcor::__internal__::create_ctrl_u();
    ctrlKernel->expand({
        std::make_pair("U", tempKernel),
        std::make_pair("control-idx", ctrlIdx),
    });

    // std::cout << "HELLO\n" << ctrlKernel->toString() << "\n";
    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      parent_kernel->addInstruction(
          ctrlKernel->getInstruction(instId)->clone());
    }
    // Need to reset and point current program to the parent
    quantum::set_current_program(parent_kernel);
  }

  // Single-qubit overload
  static void ctrl(std::shared_ptr<CompositeInstruction> parent_kernel,
                   int ctrlIdx, Args... args) {
    ctrl(parent_kernel, {ctrlIdx}, args...);
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

    // instantiate and don't let it call the destructor
    Derived derived(args...);
    derived.disable_destructor = true;

    // run the operator()(args...) call to get the the functor
    // as a CompositeInstruction (derived.parent_kernel)
    derived(args...);

    // Use the controlled gate module of XACC to transform
    auto tempKernel = qcor::__internal__::create_composite("temp_control");
    tempKernel->addInstruction(derived.parent_kernel);

    auto ctrlKernel = qcor::__internal__::create_ctrl_u();
    ctrlKernel->expand({{"U", tempKernel},
                        {"control-idx", ctrl_bits},
                        {"control-buffer", buffer_name}});

    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      parent_kernel->addInstruction(
          ctrlKernel->getInstruction(instId)->clone());
    }
    // Need to reset and point current program to the parent
    quantum::set_current_program(parent_kernel);
  }

  // Create the controlled version of this quantum kernel
  static void ctrl(std::shared_ptr<CompositeInstruction> parent_kernel,
                   qubit ctrl_qbit, Args... args) {
    int ctrl_bit = (int)ctrl_qbit.second;

    // instantiate and don't let it call the destructor
    Derived derived(args...);
    derived.disable_destructor = true;

    // run the operator()(args...) call to get the the functor
    // as a CompositeInstruction (derived.parent_kernel)
    derived(args...);

    // Use the controlled gate module of XACC to transform
    auto tempKernel = qcor::__internal__::create_composite("temp_control");
    tempKernel->addInstruction(derived.parent_kernel);

    auto ctrlKernel = qcor::__internal__::create_ctrl_u();
    ctrlKernel->expand({{"U", tempKernel},
                        {"control-idx", ctrl_bit},
                        {"control-buffer", ctrl_qbit.first}});

    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      parent_kernel->addInstruction(
          ctrlKernel->getInstruction(instId)->clone());
    }
    // Need to reset and point current program to the parent
    quantum::set_current_program(parent_kernel);
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
};

template <typename... Args>
using callable_function_ptr =
    void (*)(std::shared_ptr<xacc::CompositeInstruction>, Args...);

template <typename... Args>
class KernelSignature {
 protected:
  callable_function_ptr<Args...> &function_pointer;

 public:
  KernelSignature(callable_function_ptr<Args...> &&f) : function_pointer(f) {}
  // Ctor from raw void* funtion pointer.
  // IMPORTANT: since function_pointer is kept as a *reference*,
  // we must keep a reference to the original f_ptr void* as well.
  KernelSignature(void *&f_ptr)
      : function_pointer((callable_function_ptr<Args...> &)f_ptr) {}

  void operator()(std::shared_ptr<xacc::CompositeInstruction> ir,
                  Args... args) {
    function_pointer(ir, args...);
  }
  void ctrl(std::shared_ptr<xacc::CompositeInstruction> ir, int ctrl_qbit,
            Args... args) {
    auto tempKernel = qcor::__internal__::create_composite("temp_control");
    function_pointer(tempKernel, args...);

    auto ctrlKernel = qcor::__internal__::create_ctrl_u();
    ctrlKernel->expand({
        std::make_pair("U", tempKernel),
        std::make_pair("control-idx", ctrl_qbit),
    });

    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      ir->addInstruction(ctrlKernel->getInstruction(instId)->clone());
    }

    ::quantum::set_current_program(ir);
  }

  void ctrl(std::shared_ptr<xacc::CompositeInstruction> ir, qubit ctrl_qbit,
            Args... args) {
    int ctrl_bit = (int)ctrl_qbit.second;
    ctrl(ir, ctrl_bit, args...);
  }

  void adjoint(std::shared_ptr<CompositeInstruction> ir, Args... args) {
    auto tempKernel = qcor::__internal__::create_composite("temp_adjoint");
    function_pointer(tempKernel, args...);

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

// We use the following to enable ctrl operations on our single
// qubit gates, X::ctrl(), Z::ctrl(), H::ctrl(), etc....
template <typename Derived>
using OneQubitKernel = QuantumKernel<Derived, qubit>;

#define ONE_QUBIT_KERNEL_CTRL_ENABLER(CLASSNAME, QRTNAME)                 \
  class CLASSNAME : public OneQubitKernel<class CLASSNAME> {              \
   public:                                                                \
    CLASSNAME(qubit q) : OneQubitKernel<CLASSNAME>(q) {}                  \
    CLASSNAME(std::shared_ptr<qcor::CompositeInstruction> _parent_kernel, \
              qubit q)                                                    \
        : OneQubitKernel<CLASSNAME>(_parent_kernel, q) {                  \
      throw std::runtime_error("you cannot call this.");                  \
    }                                                                     \
    void operator()(qubit q) {                                            \
      parent_kernel = qcor::__internal__::create_composite(               \
          "__tmp_one_qubit_ctrl_enabler");                                \
      quantum::set_current_program(parent_kernel);                        \
      if (runtime_env == QrtType::FTQC) {                                 \
        quantum::set_current_buffer(q.results());                         \
      }                                                                   \
      ::quantum::QRTNAME(q);                                              \
      return;                                                             \
    }                                                                     \
    virtual ~CLASSNAME() {}                                               \
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

template <typename... Args>
using GenerateKernelBodyPtr = std::string (*)(Args...);

// This class is used to simplify the syntax of 
// passing kernel and capture variable names to the qpu_lambda
class qpu_lambda_variables {
 protected:
  std::vector<std::string> kernel_args;
  std::vector<std::string> capture_args;

 public:
  qpu_lambda_variables(std::initializer_list<std::string> k)
      : kernel_args(std::vector<std::string>(k)) {}
  qpu_lambda_variables(std::initializer_list<std::string> k,
                       std::initializer_list<std::string> c)
      : kernel_args(std::vector<std::string>(k)),
        capture_args(std::vector<std::string>(c)) {}
  std::vector<std::string> get_kernel_args() { return kernel_args; }
  std::vector<std::string> get_capture_args() { return capture_args; }
};

template <typename... Args>
class qpu_lambda {
 private:
  class TupleToTypeArgString {
   protected:
    std::string &tmp;
    std::vector<std::string> var_names;
    int counter = 0;

    template <class T>
    std::string type_name() {
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
    template <typename T>
    void operator()(T &t) {
      tmp += type_name<decltype(t)>() + " " +
             (var_names.empty() ? "arg_" + std::to_string(counter)
                                : var_names[counter]) +
             ",";
      counter++;
    }
  };

 protected:
  void *f;
  std::tuple<Args...> capture_vars;
  qpu_lambda_variables variable_names_map;

 public:
  template <typename LambdaType>
  qpu_lambda(LambdaType &&ff, qpu_lambda_variables &&variable_names,
             Args... _capture_vars)
      : variable_names_map(std::move(variable_names)),
        capture_vars(std::forward_as_tuple(_capture_vars...)) {
    f = reinterpret_cast<void *>(+ff);
  }

  template <typename... FunctionArgs>
  void operator()(FunctionArgs... args) {
    auto casted = reinterpret_cast<GenerateKernelBodyPtr<FunctionArgs...>>(f);
    std::stringstream ss;
    QJIT qjit;
    // Map the args to a tuple
    auto kernel_args_tuple = std::make_tuple(args...);

    // Get the kernel body as a string
    auto s = casted(args...);

    // Extract the kernel and capture variable names
    auto kernel_var_names = variable_names_map.get_kernel_args();
    auto capture_var_names = variable_names_map.get_capture_args();

    // Build up the function argument signature string
    std::string args_string = "";
    TupleToTypeArgString to(args_string, kernel_var_names);
    __internal__::tuple_for_each(kernel_args_tuple, to);
    TupleToTypeArgString co(args_string);
    __internal__::tuple_for_each(capture_vars, co);
    args_string = args_string.substr(0, args_string.length() - 1);

    std::string capture_preamble = "";
    for (auto [i, capture_name] : qcor::enumerate(capture_var_names)) {
      capture_preamble +=
          "auto " + capture_name + " = arg_" + std::to_string(i) + ";\n";
    }

    // Insert the capture preamble code
    s.insert(1, "\n" + capture_preamble);

    // Create the kernel string for QJIT
    ss << "__qpu__ void foo(" << args_string << ")\n" << s;

    // Compile
    qjit.jit_compile(ss.str());

    // Merge the kernel args and the capture vars and execute
    auto final_args_tuple = std::tuple_cat(kernel_args_tuple, capture_vars);
    std::apply([&](auto &&...args) { qjit.invoke("foo", args...); },
               final_args_tuple);
  }
};

#define STRINGIZE(A) #A
#define qpu_lambda_body(EXPR) return std::string(STRINGIZE(EXPR));

}  // namespace qcor
