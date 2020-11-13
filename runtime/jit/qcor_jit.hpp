#pragma once
#include <cxxabi.h>

#include <map>
#include <memory>
#include <string>

#include "heterogeneous.hpp"

namespace llvm {
class Module;
}
namespace xacc {
class CompositeInstruction;
}

namespace qcor {
class LLVMJIT;

class QJIT {
  template <typename... Args>
  using kernel_functor_t = void (*)(Args...);

 private:
  std::map<std::size_t, std::string> cached_kernel_codes;
  std::string demangle(const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  };

 protected:
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr;
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr_hetmap;
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr_parent_hetmap;

  std::unique_ptr<LLVMJIT> jit;
  std::unique_ptr<llvm::Module> module;

 public:
  QJIT();
  ~QJIT();
  const std::pair<std::string, std::string> run_syntax_handler(
      const std::string &quantum_kernel_src,
      const bool add_het_map_kernel_ctor = false);
  void jit_compile(const std::string &quantum_kernel_src,
                   const bool add_het_map_kernel_ctor = false,
                   const std::vector<std::string> &kernel_dependency = {},
                   const std::string &extra_functions_src = "");

  void write_cache();

  template <typename... Args>
  void invoke(const std::string &kernel_name, Args... args) {
    auto f_ptr = kernel_name_to_f_ptr[kernel_name];
    void (*kernel_functor)(Args...) = (void (*)(Args...))f_ptr;
    kernel_functor(args...);
  }

  void invoke_with_hetmap(const std::string &kernel_name,
                          xacc::HeterogeneousMap &args);
  std::shared_ptr<xacc::CompositeInstruction> extract_composite_with_hetmap(
      const std::string name, xacc::HeterogeneousMap &m);

  template <typename... Args>
  kernel_functor_t<Args...> get_kernel(const std::string &kernel_name) {
    auto f_ptr = kernel_name_to_f_ptr[kernel_name];
    void (*kernel_functor)(Args...) = (void (*)(Args...))f_ptr;
    return kernel_functor;
  }
};

}  // namespace qcor