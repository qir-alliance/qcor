#include "qir-qrt.hpp"
#include <iostream>

extern "C" {
void __quantum__rt__callable_update_reference_count(Callable *clb, int32_t c) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__rt__callable_update_alias_count(Callable *clb, int32_t c) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__rt__callable_invoke(Callable *clb, TuplePtr args,
                                    TuplePtr res) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  if (clb) {
    clb->invoke(args, res);
  }
  else {
    std::cout << "Callback is NULL.\n";
  }
}
Callable *__quantum__rt__callable_copy(Callable *clb, bool force) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}
void __quantum__rt__capture_update_reference_count(Callable *clb,
                                                   int32_t count) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__rt__capture_update_alias_count(Callable *clb, int32_t count) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
void __quantum__rt__callable_memory_management(int32_t index, Callable *clb,
                                               int64_t parameter) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
}