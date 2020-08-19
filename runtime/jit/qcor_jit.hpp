#include "qalloc.hpp"
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

namespace llvm {
class Module;
}
namespace qcor {
class LLVMJIT;

class QJIT {

protected:
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr;

  std::unique_ptr<LLVMJIT> jit;
  std::unique_ptr<llvm::Module> module;

public:
  QJIT();
  ~QJIT();
  const std::pair<std::string, std::string>
  run_syntax_handler(const std::string &quantum_kernel_src);
  void jit_compile(const std::string &quantum_kernel_src);

  template <typename... Args>
  void invoke(const std::string &kernel_name, Args... args) {
    auto f_ptr = kernel_name_to_f_ptr[kernel_name];
    void (*kernel_functor)(Args...) = (void (*)(Args...))f_ptr;
    kernel_functor(args...);
  }
};


} // namespace qcor