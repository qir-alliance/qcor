#include "qir-qrt.hpp"
#include <iostream>

extern "C" {
TuplePtr __quantum__rt__tuple_create(int64_t size) {
  if (verbose)
    std::cout << "[qir-qrt] Create a tuple of size " << size << ".\n";
  auto tupleHeader = TupleHeader::create(size);
  return tupleHeader->getTuple();
}

void __quantum__rt__tuple_update_reference_count(int8_t *tuple,
                                                 int32_t increment) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}

void __quantum__rt__tuple_update_alias_count(int8_t *tuple, int32_t increment) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}

TuplePtr __quantum__rt__tuple_copy(int8_t *tuple, bool forceNewInstance) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return nullptr;
}

void Callable::invoke(TuplePtr args, TuplePtr result) {
  std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  if (m_functor) {
    m_functor->execute(args, result);
  }
}
}