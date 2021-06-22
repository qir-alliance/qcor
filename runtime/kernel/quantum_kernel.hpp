#pragma once
#include <optional>
#include <stack>

#include "qcor_jit.hpp"
#include "qcor_observable.hpp"
#include "qcor_utils.hpp"
#include "qrt.hpp"

namespace qcor {
enum class QrtType { NISQ, FTQC };

// Forward declare
template <typename... Args>
class KernelSignature;

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
double observe(Operator &obs, KernelSignature<Args...> &kernelCallable,
               Args... args);

template <typename... Args>
UnitaryMatrix as_unitary_matrix(KernelSignature<Args...> &kernelCallable,
                                Args... args);
template <typename... Args>
std::string openqasm(KernelSignature<Args...> &kernelCallable, Args... args);

template <typename... Args>
void print_kernel(KernelSignature<Args...> &kernelCallable, std::ostream &os,
                  Args... args);

template <typename... Args>
std::size_t n_instructions(KernelSignature<Args...> &kernelCallable,
                           Args... args);
}  // namespace internal

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
    KernelSignature<Args...> callable(derived);
    return internal::print_kernel(callable, os, args...);
  }

  static void print_kernel(Args... args) { print_kernel(std::cout, args...); }

  // Static method to query how many instructions are in this kernel
  static std::size_t n_instructions(Args... args) {
    Derived derived(args...);
    KernelSignature<Args...> callable(derived);
    return internal::n_instructions(callable, args...);
  }

  // Create the Adjoint of this quantum kernel
  static void adjoint(std::shared_ptr<CompositeInstruction> parent_kernel,
                      Args... args) {
    Derived derived(args...);
    KernelSignature<Args...> callable(derived);
    return internal::apply_adjoint(parent_kernel, callable, args...);
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

  static UnitaryMatrix as_unitary_matrix(Args... args) {
    Derived derived(args...);
    KernelSignature<Args...> callable(derived);
    return internal::as_unitary_matrix(callable, args...);
  }

  static double observe(Operator &obs, Args... args) {
    Derived derived(args...);
    KernelSignature<Args...> callable(derived);
    return internal::observe(obs, callable, args...);
  }

  static double observe(std::shared_ptr<Operator> obs, Args... args) {
    return observe(*obs, args...);
  }

  // Simple autograd support for kernel with simple type: double or
  // vector<double>. Other signatures must provide a translator...
  // static double autograd(Operator &obs, std::vector<double> &dx, qreg q,
  //                        double x) {
  //   std::function<std::shared_ptr<CompositeInstruction>(
  //       std::vector<double>)>
  //       kernel_eval = [q](std::vector<double> x) {
  //         auto tempKernel =
  //             qcor::__internal__::create_composite("__temp__autograd__");
  //         Derived derived(q, x[0]);
  //         derived.disable_destructor = true;
  //         derived(q, x[0]);
  //         tempKernel->addInstructions(derived.parent_kernel->getInstructions());
  //         return tempKernel;
  //       };

  //   auto gradiend_method = qcor::__internal__::get_gradient_method(
  //       qcor::__internal__::DEFAULT_GRADIENT_METHOD, kernel_eval, obs);
  //   const double cost_val = observe(obs, q, x);
  //   dx = (*gradiend_method)({x}, cost_val);
  //   return cost_val;
  // }

  // static double autograd(Operator &obs, std::vector<double> &dx, qreg q,
  //                        std::vector<double> x) {
  //   std::function<std::shared_ptr<CompositeInstruction>(
  //       std::vector<double>)>
  //       kernel_eval = [q](std::vector<double> x) {
  //         auto tempKernel =
  //             qcor::__internal__::create_composite("__temp__autograd__");
  //         Derived derived(q, x);
  //         derived.disable_destructor = true;
  //         derived(q, x);
  //         tempKernel->addInstructions(derived.parent_kernel->getInstructions());
  //         return tempKernel;
  //       };

  //   auto gradiend_method = qcor::__internal__::get_gradient_method(
  //       qcor::__internal__::DEFAULT_GRADIENT_METHOD, kernel_eval, obs);
  //   const double cost_val = observe(obs, q, x);
  //   dx = (*gradiend_method)(x, cost_val);
  //   return cost_val;
  // }

  // static double autograd(Operator &obs, std::vector<double> &dx,
  //                        std::vector<double> x,
  //                        ArgsTranslator<Args...> args_translator) {
  //   std::function<std::shared_ptr<CompositeInstruction>(
  //       std::vector<double>)>
  //       kernel_eval = [&](std::vector<double> x_vec) {
  //         auto eval_lambda = [&](Args... args) {
  //           auto tempKernel =
  //               qcor::__internal__::create_composite("__temp__autograd__");
  //           Derived derived(args...);
  //           derived.disable_destructor = true;
  //           derived(args...);
  //           tempKernel->addInstructions(
  //               derived.parent_kernel->getInstructions());
  //           return tempKernel;
  //         };
  //         auto args_tuple = args_translator(x_vec);
  //         return std::apply(eval_lambda, args_tuple);
  //       };

  //   auto gradiend_method = qcor::__internal__::get_gradient_method(
  //       qcor::__internal__::DEFAULT_GRADIENT_METHOD, kernel_eval, obs);

  //   auto kernel_observe = [&](Args... args) { return observe(obs, args...);
  //   };

  //   auto args_tuple = args_translator(x);
  //   const double cost_val = std::apply(kernel_observe, args_tuple);
  //   dx = (*gradiend_method)(x, cost_val);
  //   return cost_val;
  // }

  static std::string openqasm(Args... args) {
    Derived derived(args...);
    KernelSignature<Args...> callable(derived);
    return internal::openqasm(callable, args...);
  }

  virtual ~QuantumKernel() {}

  template <typename... ArgTypes>
  friend class KernelSignature;
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

template <typename... CaptureArgs>
class _qpu_lambda {
 private:
  // Private inner class for getting the type
  // of a capture variable as a string at runtime
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
  // Variational information, i.e. is this lambda compatible with VQE
  // e.g. single double or single vector double input.
  enum class Variational_Arg_Type { Double, Vec_Double, None };
  Variational_Arg_Type var_type = Variational_Arg_Type::None;

  // Constructor, capture vars should be deduced without
  // specifying them since we're using C++17
  _qpu_lambda(std::string &&ff, std::string &&_capture_var_names,
              CaptureArgs &..._capture_vars)
      : src_str(ff),
        capture_var_names(_capture_var_names),
        capture_vars(std::forward_as_tuple(_capture_vars...)) {
    // Get the original args list
    auto first = src_str.find_first_of("(");
    auto last = src_str.find_first_of(")");
    auto tt = src_str.substr(first, last - first + 1);
    // Parse the argument list
    const auto arg_type_and_names = [](const std::string &arg_string_decl)
        -> std::vector<std::pair<std::string, std::string>> {
      // std::cout << "HOWDY:" << arg_string_decl << "\n";
      std::vector<std::pair<std::string, std::string>> result;
      const auto args_string =
          arg_string_decl.substr(1, arg_string_decl.size() - 2);
      std::stack<char> grouping_chars;
      std::string type_name;
      std::string var_name;
      std::string temp;
      // std::cout << args_string << "\n";
      for (int i = 0; i < args_string.size(); ++i) {
        if (isspace(args_string[i]) && grouping_chars.empty()) {
          type_name = temp;
          temp.clear();
        } else if (args_string[i] == ',') {
          var_name = temp;
          if (var_name[0] == '&') {
            type_name += "&";
            var_name = var_name.substr(1);
          }
          result.emplace_back(std::make_pair(type_name, var_name));
          type_name.clear();
          var_name.clear();
          temp.clear();
        } else {
          temp.push_back(args_string[i]);
        }

        if (args_string[i] == '<') {
          grouping_chars.push(args_string[i]);
        }
        if (args_string[i] == '>') {
          // assert(grouping_chars.top() == '<');
          grouping_chars.pop();
        }
      }

      // Last one:
      var_name = temp;
      if (var_name[0] == '&') {
        type_name += "&";
        var_name = var_name.substr(1);
      }
      result.emplace_back(std::make_pair(type_name, var_name));
      return result;
    }(tt);

    // Determine if this lambda has a VQE-compatible type:
    // QReg then variational params.
    if (arg_type_and_names.size() == 2) {
      const auto trim_space = [](std::string &stripString) {
        while (!stripString.empty() && std::isspace(*stripString.begin())) {
          stripString.erase(stripString.begin());
        }

        while (!stripString.empty() && std::isspace(*stripString.rbegin())) {
          stripString.erase(stripString.length() - 1);
        }
      };

      auto type_name = arg_type_and_names[1].first;
      trim_space(type_name);
      // Use a relax search to handle using namespace std...
      // FIXME: this is quite hacky.
      if (type_name.find("vector<double>") != std::string::npos) {
        var_type = Variational_Arg_Type::Vec_Double;
      } else if (type_name == "double") {
        var_type = Variational_Arg_Type::Double;
      }
    }

    // Map simple type to its reference type so that the
    // we can use consistent type-forwarding
    // when casting the JIT raw function pointer.
    // Currently, looks like only these simple types are having problem
    // with perfect type forwarding.
    // i.e. by-value arguments of these types are incompatible with a by-ref
    // casted function.
    static const std::unordered_map<std::string, std::string>
        FORWARD_TYPE_CONVERSION_MAP{{"int", "int&"}, {"double", "double&"}};
    std::vector<std::pair<std::string, std::string>> forward_types;
    // Replicate by-value by create copies and restore the variables.
    std::vector<std::string> byval_casted_arg_names;
    for (const auto &[type, name] : arg_type_and_names) {
      // std::cout << type << " --> " << name << "\n";
      if (FORWARD_TYPE_CONVERSION_MAP.find(type) !=
          FORWARD_TYPE_CONVERSION_MAP.end()) {
        auto iter = FORWARD_TYPE_CONVERSION_MAP.find(type);
        forward_types.emplace_back(std::make_pair(iter->second, name));
        byval_casted_arg_names.emplace_back(name);
      } else {
        forward_types.emplace_back(std::make_pair(type, name));
      }
    }

    // std::cout << "After\n";
    // Construct the new arg signature clause:
    std::string arg_clause_new;
    arg_clause_new.push_back('(');
    for (const auto &[type, name] : forward_types) {
      arg_clause_new.append(type);
      arg_clause_new.push_back(' ');
      arg_clause_new.append(name);
      arg_clause_new.push_back(',');
      // std::cout << type << " --> " << name << "\n";
    }
    arg_clause_new.pop_back();

    // Get the capture type:
    // By default "[]", pass by reference.
    // [=]: pass by value
    // [&]: pass by reference (same as default)
    const auto first_square_bracket = src_str.find_first_of("[");
    const auto last_square_bracket = src_str.find_first_of("]");
    const auto capture_type = src_str.substr(
        first_square_bracket, last_square_bracket - first_square_bracket + 1);
    if (!capture_type.empty() && capture_type == "[=]") {
      // We must check this at compile-time to prevent the compiler from
      // *attempting* to compile this code path even when by-val capture is not
      // in use. The common scenario is a qpu_lambda captures other qpu_lambda.
      // Copying of qpu_lambda by value is prohibitied.
      // We'll report a runtime error for this case.
      if constexpr (std::conjunction_v<
                        std::is_copy_assignable<CaptureArgs>...>) {
        // Store capture vars (by-value)
        optional_copy_capture_vars = std::forward_as_tuple(_capture_vars...);
      } else {
        error(
            "Capture variable type is non-copyable. Cannot use capture by "
            "value.");
      }
    }

    // Need to append capture vars to this arg signature
    std::string capture_preamble = "";
    const auto replaceVarName = [](std::string &str, const std::string &from,
                                   const std::string &to) {
      size_t start_pos = str.find(from);
      if (start_pos != std::string::npos) {
        str.replace(start_pos, from.length(), to);
      }
    };
    if (!capture_var_names.empty()) {
      std::string args_string = "";
      TupleToTypeArgString co(args_string);
      __internal__::tuple_for_each(capture_vars, co);
      args_string = "," + args_string.substr(0, args_string.length() - 1);

      // Replace the generic argument names (tuple foreach)
      // with the actual capture var name.
      // We need to do this so that the SyntaxHandler can properly detect if
      // a capture var is a Kernel-like ==> add the list of in-flight kernels
      // and add parent_kernel to the invocation.
      for (auto [i, capture_name] :
           qcor::enumerate(qcor::split(capture_var_names, ','))) {
        const auto old_name = "arg_" + std::to_string(i);
        replaceVarName(args_string, old_name, capture_name);
      }

      tt.insert(last - capture_type.size(), args_string);
      arg_clause_new.append(args_string);
    }

    // Extract the function body
    first = src_str.find_first_of("{");
    last = src_str.find_last_of("}");
    auto rr = src_str.substr(first, last - first + 1);
    arg_clause_new.push_back(')');
    // std::cout << "New signature: " << arg_clause_new << "\n";
    // Reconstruct with new args signature and
    // existing function body
    std::stringstream ss;
    ss << "__qpu__ void foo" << arg_clause_new << rr;

    // Get as a string, and insert capture
    // preamble if necessary
    auto jit_src = ss.str();
    first = jit_src.find_first_of("{");
    if (!capture_var_names.empty()) jit_src.insert(first + 1, capture_preamble);

    if (!byval_casted_arg_names.empty()) {
      std::stringstream cache_string, restore_string;
      for (const auto &var : byval_casted_arg_names) {
        cache_string << "auto __" << var << "__cached__ = " << var << ";\n";
        restore_string << var << " = __" << var << "__cached__;\n";
      }
      const auto begin = jit_src.find_first_of("{");
      jit_src.insert(begin + 1, cache_string.str());
      const auto end = jit_src.find_last_of("}");
      jit_src.insert(end, restore_string.str());
    }

    std::cout << "JITSRC:\n" << jit_src << "\n";
    // JIT Compile, storing the function pointers
    qjit.jit_compile(jit_src);
  }

  template <typename... FunctionArgs>
  void eval_with_parent(std::shared_ptr<CompositeInstruction> parent,
                        FunctionArgs &&...args) {
    this->operator()(parent, std::forward<FunctionArgs>(args)...);
  }

  template <typename... FunctionArgs>
  void operator()(std::shared_ptr<CompositeInstruction> parent,
                  FunctionArgs &&...args) {
    // Map the function args to a tuple
    auto kernel_args_tuple = std::forward_as_tuple(args...);

    if (!optional_copy_capture_vars.has_value()) {
      // By-ref:
      // Merge the function args and the capture vars and execute
      auto final_args_tuple = std::tuple_cat(kernel_args_tuple, capture_vars);
      std::apply(
          [&](auto &&...args) {
            qjit.invoke_with_parent_forwarding("foo", parent, args...);
          },
          final_args_tuple);
    } else if constexpr (std::conjunction_v<
                             std::is_copy_assignable<CaptureArgs>...>) {
      // constexpr compile-time check to prevent compiler from looking at this
      // code path if the capture variable is non-copyable, e.g. qpu_lambda.
      // By-value:
      auto final_args_tuple =
          std::tuple_cat(kernel_args_tuple, optional_copy_capture_vars.value());
      std::apply(
          [&](auto &&...args) {
            qjit.invoke_with_parent_forwarding("foo", parent, args...);
          },
          final_args_tuple);
    }
  }

  template <typename... FunctionArgs>
  void operator()(FunctionArgs &&...args) {
    // Map the function args to a tuple
    auto kernel_args_tuple = std::forward_as_tuple(args...);
    if (!optional_copy_capture_vars.has_value()) {
      // By-ref
      // Merge the function args and the capture vars and execute
      auto final_args_tuple = std::tuple_cat(kernel_args_tuple, capture_vars);
      std::apply(
          [&](auto &&...args) { qjit.invoke_forwarding("foo", args...); },
          final_args_tuple);
    } else if constexpr (std::conjunction_v<
                             std::is_copy_assignable<CaptureArgs>...>) {
      // By-value
      auto final_args_tuple =
          std::tuple_cat(kernel_args_tuple, optional_copy_capture_vars.value());
      std::apply(
          [&](auto &&...args) { qjit.invoke_forwarding("foo", args...); },
          final_args_tuple);
    }
  }

  template <typename... FunctionArgs>
  double observe(std::shared_ptr<Operator> obs, FunctionArgs... args) {
    return observe(*obs.get(), args...);
  }

  template <typename... FunctionArgs>
  double observe(Operator &obs, FunctionArgs... args) {
    KernelSignature<FunctionArgs...> callable(*this);
    return internal::observe(obs, callable, args...);
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
      ctrl_qubit_vec.push_back(
          {"q", static_cast<size_t>(ctrl_idxs[i]), nullptr});
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

  template <typename... FunctionArgs>
  void adjoint(std::shared_ptr<CompositeInstruction> parent_kernel,
               FunctionArgs... args) {
    KernelSignature<FunctionArgs...> callable(*this);
    return internal::apply_adjoint(parent_kernel, callable, args...);
  }

  template <typename... FunctionArgs>
  void print_kernel(std::ostream &os, FunctionArgs... args) {
    KernelSignature<FunctionArgs...> callable(*this);
    return internal::print_kernel(callable, os, args...);
  }

  template <typename... FunctionArgs>
  void print_kernel(FunctionArgs... args) {
    print_kernel(std::cout, args...);
  }

  template <typename... FunctionArgs>
  std::size_t n_instructions(FunctionArgs... args) {
    KernelSignature<FunctionArgs...> callable(*this);
    return internal::n_instructions(callable, args...);
  }

  template <typename... FunctionArgs>
  UnitaryMatrix as_unitary_matrix(FunctionArgs... args) {
    KernelSignature<FunctionArgs...> callable(*this);
    return internal::as_unitary_matrix(callable, args...);
  }

  template <typename... FunctionArgs>
  std::string openqasm(FunctionArgs... args) {
    KernelSignature<FunctionArgs...> callable(*this);
    return internal::openqasm(callable, args...);
  }
};

#define qpu_lambda(EXPR, ...) _qpu_lambda(#EXPR, #__VA_ARGS__, ##__VA_ARGS__)

template <typename... Args>
using callable_function_ptr = void (*)(std::shared_ptr<CompositeInstruction>,
                                       Args...);

template <typename... Args>
class KernelSignature {
 private:
  callable_function_ptr<Args...> *readOnly = 0;
  callable_function_ptr<Args...> &function_pointer;
  std::function<void(std::shared_ptr<CompositeInstruction>, Args...)>
      lambda_func;
  std::shared_ptr<CompositeInstruction> parent_kernel;

 public:
  // Here we set function_pointer to null and instead
  // only use lambda_func. If we set lambda_func, function_pointer
  // will never be used, so we should be good.
  template <typename... CaptureArgs>
  KernelSignature(_qpu_lambda<CaptureArgs...> &lambda)
      : function_pointer(*readOnly),
        lambda_func([&](std::shared_ptr<CompositeInstruction> pp, Args... a) {
          lambda(pp, a...);
        }) {}

  KernelSignature(callable_function_ptr<Args...> &&f) : function_pointer(f) {}

  // CTor from a QCOR QuantumKernel instance:
  template <
      typename KernelType,
      std::enable_if_t<
          std::is_base_of_v<QuantumKernel<KernelType, Args...>, KernelType>,
          bool> = true>
  KernelSignature(KernelType &kernel)
      : function_pointer(*readOnly),
        lambda_func([&](std::shared_ptr<CompositeInstruction> pp, Args... a) {
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

  void operator()(std::shared_ptr<CompositeInstruction> ir, Args... args) {
    if (lambda_func) {
      lambda_func(ir, args...);
      return;
    }

    function_pointer(ir, args...);
  }

  void operator()(Args... args) { operator()(parent_kernel, args...); }

  void set_parent_kernel(std::shared_ptr<CompositeInstruction> ir) {
    parent_kernel = ir;
  }

  void ctrl(std::shared_ptr<CompositeInstruction> ir,
            const std::vector<qubit> &ctrl_qbits, Args... args) {
    internal::apply_control(ir, ctrl_qbits, *this, args...);
  }

  void ctrl(std::shared_ptr<CompositeInstruction> ir,
            const std::vector<int> ctrl_idxs, Args... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrl_idxs.size(); i++) {
      ctrl_qubit_vec.push_back(
          {"q", static_cast<size_t>(ctrl_idxs[i]), nullptr});
    }
    ctrl(ir, ctrl_qubit_vec, args...);
  }
  void ctrl(std::shared_ptr<CompositeInstruction> ir, int ctrl_qbit,
            Args... args) {
    ctrl(ir, std::vector<int>{ctrl_qbit}, args...);
  }

  void ctrl(std::shared_ptr<CompositeInstruction> ir, qubit ctrl_qbit,
            Args... args) {
    ctrl(ir, std::vector<qubit>{ctrl_qbit}, args...);
  }

  void ctrl(std::shared_ptr<CompositeInstruction> ir, qreg ctrl_qbits,
            Args... args) {
    std::vector<qubit> ctrl_qubit_vec;
    for (int i = 0; i < ctrl_qbits.size(); i++) {
      ctrl_qubit_vec.push_back(ctrl_qbits[i]);
    }
    ctrl(ir, ctrl_qubit_vec, args...);
  }

  void adjoint(std::shared_ptr<CompositeInstruction> ir, Args... args) {
    internal::apply_adjoint(ir, *this, args...);
  }

  void print_kernel(std::ostream &os, Args... args) {
    return internal::print_kernel(*this, os, args...);
  }

  void print_kernel(Args... args) { print_kernel(std::cout, args...); }

  std::size_t n_instructions(Args... args) {
    return internal::n_instructions(*this, args...);
  }

  UnitaryMatrix as_unitary_matrix(Args... args) {
    return internal::as_unitary_matrix(*this, args...);
  }

  std::string openqasm(Args... args) {
    return internal::openqasm(*this, args...);
  }

  double observe(std::shared_ptr<Operator> obs, Args... args) {
    return observe(*obs.get(), args...);
  }

  double observe(Operator &obs, Args... args) {
    return internal::observe(obs, *this, args...);
  }
};

// Templated helper to attach parent_kernel to any
// KernelSignature arguments even nested in a std::vector<KernelSignature>
// The reason is that the Token Collector relies on a list of kernel names
// in the translation unit to attach parent_kernel to the operator() call.
// For KernelSignature provided in a container, tracking these at the
// TokenCollector level is error-prone (e.g. need to track any array accesses).
// Hence, we iterate over all kernel arguments and attach the parent_kernel
// to any KernelSignature argument at the top of each kernel's operator() call
// in a type-safe manner.

// Last arg
inline void init_kernel_signature_args_impl(
    std::shared_ptr<CompositeInstruction> ir) {}
template <typename T, typename... ArgsType>
void init_kernel_signature_args_impl(std::shared_ptr<CompositeInstruction> ir,
                                     T &t, ArgsType &...Args);

// Main function: to be added by the token collector at the beginning
// of each kernel operator().
template <typename... T>
void init_kernel_signature_args(std::shared_ptr<CompositeInstruction> ir,
                                T &...multi_inputs) {
  init_kernel_signature_args_impl(ir, multi_inputs...);
}

// Base case: generic type T,
// just ignore, proceed to the next arg.
template <typename T, typename... ArgsType>
void init_kernel_signature_args_impl(std::shared_ptr<CompositeInstruction> ir,
                                     T &t, ArgsType &...Args) {
  init_kernel_signature_args(ir, Args...);
}

// Special case: this is a vector:
// iterate over all elements.
template <typename T, typename... ArgsType>
void init_kernel_signature_args_impl(std::shared_ptr<CompositeInstruction> ir,
                                     std::vector<T> &vec_arg,
                                     ArgsType... Args) {
  for (auto &el : vec_arg) {
    // Iterate the vector elements.
    init_kernel_signature_args_impl(ir, el);
  }
  // Proceed with the rest.
  init_kernel_signature_args(ir, Args...);
}

// Handle KernelSignature arg => set the parent kernel.
template <typename... ArgsType>
void init_kernel_signature_args_impl(
    std::shared_ptr<CompositeInstruction> ir,
    KernelSignature<ArgsType...> &kernel_signature) {
  kernel_signature.set_parent_kernel(ir);
}

namespace internal {
// KernelSignature is the base of all kernel-like objects
// and we use it to implement kernel modifiers && utilities.
// Make this a utility function so that implicit conversion to KernelSignature
// occurs automatically.
template <typename... Args>
void apply_control(std::shared_ptr<CompositeInstruction> parent_kernel,
                   const std::vector<qubit> &ctrl_qbits,
                   KernelSignature<Args...> &kernelCallable, Args... args) {
  std::vector<std::pair<std::string, size_t>> ctrl_qubits;
  for (const auto &qb : ctrl_qbits) {
    ctrl_qubits.emplace_back(std::make_pair(qb.first, qb.second));
  }

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

  auto ctrlKernel = qcor::__internal__::create_and_expand_ctrl_u(
      {{"U", tempKernel}, {"control-idx", ctrl_qubits}});
  // ctrlKernel->expand({{"U", tempKernel}, {"control-idx", ctrl_qubits}});

  // Mark all the *Controlled* instructions as compute segment
  // if it was in the compute_section.
  // i.e. we have bypassed the marker previously to make C-U to work,
  // now we mark all the generated instructions.
  if (cached_is_compute_section) {
    for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
      ctrlKernel->attachMetadata(instId,
                                 {{"__qcor__compute__segment__", true}});
    }
  }

  for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
    parent_kernel->addInstruction(ctrlKernel->getInstruction(instId));
  }
  // Need to reset and point current program to the parent
  quantum::set_current_program(parent_kernel);
}

bool is_not_measure(std::shared_ptr<xacc::Instruction> inst);
std::vector<std::shared_ptr<xacc::Instruction>> handle_adjoint_instructions(
    std::vector<std::shared_ptr<xacc::Instruction>>,
    std::shared_ptr<CompositeInstruction>);

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
          [](const auto &inst) { return is_not_measure(inst); })) {
    error("Unable to create Adjoint for kernels that have Measure operations.");
  }

  auto new_instructions = internal::handle_adjoint_instructions(instructions, tempKernel);

  // add the instructions to the current parent kernel
  parent_kernel->addInstructions(std::move(new_instructions), false);

  ::quantum::set_current_program(parent_kernel);
}

template <typename... Args>
double observe(Operator &obs, KernelSignature<Args...> &kernelCallable,
               Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_observe");
  kernelCallable(tempKernel, args...);
  auto instructions = tempKernel->getInstructions();
  // Assert that we don't have measurement
  if (!std::all_of(instructions.cbegin(), instructions.cend(),
                   [](const auto &inst) { return is_not_measure(inst); })) {
    error("Unable to observe kernels that already have Measure operations.");
  }

  xacc::internal_compiler::execute_pass_manager();

  // Operator pre-processing, map down to Pauli and cache
  Operator observable;
  auto obs_str = obs.toString();
  std::hash<std::string> hasher;
  auto operator_hash = hasher(obs_str);
  if (__internal__::cached_observables.count(operator_hash)) {
    observable = __internal__::cached_observables[operator_hash];
  } else {
    if (obs_str.find("^") != std::string::npos) {
      error("We have not implemented the case where the operator is Fermion...");
      // FIXME HANDLE THIS
      //   try {
      //     observable =
      //         operatorTransform("jw", dynamic_cast<FermionOperator &>(obs));
      //   } catch (std::exception &ex) {
      //     auto fermionObservable = createOperator("fermion", obs_str);
      //     observable = operatorTransform("jw", fermionObservable);
      //   }

      //   // observable is PauliOperator, but does not cast down to it
      //   // Not sure about the likelihood of this happening, but want to cover
      //   all
      //   // bases
    } else if (obs_str.find("X") != std::string::npos ||
               obs_str.find("Y") != std::string::npos ||
               obs_str.find("Z") != std::string::npos) {
      observable = createOperator("pauli", obs_str);
    }
    __internal__::cached_observables.insert({operator_hash, observable});
  }

  // Will fail to compile if more than one qreg is passed.
  std::tuple<Args...> tmp(std::forward_as_tuple(args...));
  auto q = std::get<qreg>(tmp);
  return qcor::observe(tempKernel, observable, q);
}

template <typename... Args>
UnitaryMatrix as_unitary_matrix(KernelSignature<Args...> &kernelCallable,
                                Args... args) {
  auto tempKernel = qcor::__internal__::create_composite("temp_as_unitary");
  kernelCallable(tempKernel, args...);
  auto instructions = tempKernel->getInstructions();
  // Assert that we don't have measurement
  if (!std::all_of(
          instructions.cbegin(), instructions.cend(),
          [](const auto &inst) { return is_not_measure(inst); })) {
    error(
        "Unable to compute unitary matrix for kernels that already have "
        "Measure operations.");
  }

  return __internal__::map_composite_to_unitary_matrix(tempKernel);
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
}  // namespace internal
}  // namespace qcor
