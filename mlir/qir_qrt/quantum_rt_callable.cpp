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
  if (clb == nullptr) {
    return nullptr;
  }
  auto clone = new Callable(*clb);
  return clone;
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

void __quantum__rt__callable_make_adjoint(Callable *clb) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  if (clb == nullptr) {
    return;
  }
  clb->applyFunctor(Callable::AdjointIdx);
}
void __quantum__rt__callable_make_controlled(Callable *clb) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  if (clb == nullptr) {
    return;
  }
  clb->applyFunctor(Callable::ControlledIdx);
}
Callable *
__quantum__rt__callable_create(Callable::CallableEntryType *ft,
                               Callable::CaptureCallbackType *callbacks,
                               TuplePtr capture) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  auto clb = new Callable(ft, callbacks, capture);
  return clb;
}
}