#include "qir-qrt.hpp"
#include <iostream>

extern "C" {
int8_t *__quantum__rt__tuple_create(int64_t size) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Leak..
  Array *tuplePtrArray = new Array(size);
  return reinterpret_cast<int8_t *>(tuplePtrArray);
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

int8_t *__quantum__rt__tuple_copy(int8_t *tuple, bool forceNewInstance) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  return nullptr;
}
}