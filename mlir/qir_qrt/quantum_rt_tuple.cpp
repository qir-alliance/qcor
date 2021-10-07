/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "qir-qrt.hpp"
#include <iostream>

namespace {
// Tuple with a particular structure  
struct TupleWithControls {
  Array *controls;
  TupleWithControls *innerTuple;
};
TuplePtr flattenControlArrays(TuplePtr tuple, int depth) {
  if (depth == 1) {
    return tuple;
  }

  const size_t qubitSize = sizeof(Qubit *);
  // Peel off each layer of control:
  // Assuming the TupleWithControls structure (specific to Q#)
  Array *combinedControls = new Array(0, qubitSize);
  TupleWithControls *current = reinterpret_cast<TupleWithControls *>(tuple);
  // Remaining arguments
  TupleHeader *last = nullptr;
  for (int i = 0; i < depth; i++) {
    if (i == depth - 1) {
      last = TupleHeader::getHeader(reinterpret_cast<TuplePtr>(current));
    }
    // Get control array
    Array *controls = current->controls;
    combinedControls->append(*controls);
    current = current->innerTuple;
  }

  TupleHeader *flatTuple = TupleHeader::create(last);
  Array **arr = reinterpret_cast<Array **>(flatTuple->getTuple());
  *arr = combinedControls;
  // std::cout << "Combined controls of size = " << combinedControls->size() << "\n";
  return flatTuple->getTuple();
}
} // namespace
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
      auto flatTuple = flattenControlArrays(args, m_controlledDepth);
      m_functionTable[m_functorIdx](m_capture, flatTuple, result);
    }
  }
}
}