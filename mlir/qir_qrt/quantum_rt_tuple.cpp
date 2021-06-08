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
  if (!tuple) {
    // Ignored
    return;
  }

  auto tuple_header = TupleHeader::getHeader(tuple);

  if (increment > 0) {
    for (int64_t i = 0; i < increment; ++i) {
      tuple_header->add_ref();
    }
  } else {
    for (int64_t i = 0; i < (-increment); ++i) {
      if (tuple_header->release_ref()) {
        // The tuple has been deallocated
        if (verbose)
          std::cout << "Deallocates tuple.\n";
        return;
      }
    }
  }
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
  if (m_functor) {
    m_functor->execute(args, result);
    return;
  }
  if (m_functionTable[m_functorIdx]) {
    if (m_controlledDepth < 2) {
      m_functionTable[m_functorIdx](m_capture, args, result);
    }
    else {
      // TODO: flatten the control array.
      throw "Multi-controlled is not supported yet.";
    }
  }
}
}