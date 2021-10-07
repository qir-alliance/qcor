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
#include <cstring>
extern "C" {
Array *__quantum__rt__array_create_1d(int32_t itemSizeInBytes,
                                      int64_t count_items) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return new Array(count_items, itemSizeInBytes);
}

int64_t __quantum__rt__array_get_size_1d(Array *state1) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return state1->size();
}

Array *__quantum__rt__array_slice(Array *array, int32_t dim,
                                  int64_t range_start, int64_t range_step,
                                  int64_t range_end) {
  return quantum__rt__array_slice(array, dim,
                                  {range_start, range_step, range_end});
}

Array *__quantum__rt__array_slice_1d(Array *array, int64_t range_start,
                                     int64_t range_step, int64_t range_end) {
  return __quantum__rt__array_slice(array, 0, range_start, range_step,
                                    range_end);
}

Array *quantum__rt__array_slice(Array *array, int32_t dim, Range range) {
  if (verbose)
    std::cout << "[qir-qrt] Extract array slice (dim = " << dim
              << ") for the range [" << range.start << ":" << range.step << ":"
              << range.end << "].\n";

  const std::vector<int64_t> range_idxs = qcor::getRangeValues(array, range);
  if (verbose) {
    std::cout << "[qir-qrt] Resolve range indices:";
    for (const auto &idx : range_idxs) {
      std::cout << idx << " ";
    }
    std::cout << "\n";
  }

  auto resultArray = new Array(range_idxs.size(), array->element_size());
  int64_t counter = 0;
  for (const auto &idx : range_idxs) {
    int8_t *src_ptr = array->getItemPointer(idx);
    int8_t *dest_ptr = resultArray->getItemPointer(counter);
    memcpy(dest_ptr, src_ptr, array->element_size());
    counter++;
  }

  return resultArray;
}

void __quantum__rt__array_update_alias_count(Array *array, int32_t increment) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // Looks like alias count has no functional significance, hence ignored.
}

void __quantum__rt__array_update_reference_count(Array *array, int32_t increment) {
  // Spec:
  // Deallocates the array if the reference count becomes 0. 
  // The behavior is undefined if the reference count becomes negative. 
  // The call should be ignored if the given %Array* is a null pointer.  
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  if (!array) {
    // The call should be ignored if the given %Array* is a null pointer.
    return;
  }

  if (verbose)
    std::cout << "Current ref. count = " << array->ref_count() << "; increment = " << increment << "\n";
  
  if (increment > 0) {
    for (int64_t i = 0; i < increment; ++i) {
      array->add_ref();
    }
  } else {
    for (int64_t i = 0; i < (-increment); ++i) {
      if (array->release_ref()) {
        // Deallocates the array if the reference count becomes 0. 
        if (verbose)
          std::cout << "Deallocates array.\n";
        delete array;
        return;
      }
    }
  }
}

// Returns the number of dimensions in the array.
int32_t __quantum__rt__array_get_dim(Array *array) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return 1;
}

int64_t __quantum__rt__array_get_size(Array *array, int32_t dim) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // We don't support multi-dimensional arrays (yet).
  return 0;
}

Array *__quantum__rt__array_copy(Array *array, bool forceNewInstance) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";

  if (array && forceNewInstance) {
    return new Array(*array);
  }
  // Spec:
  // Returns the given array pointer (the first parameter), 
  // after increasing its reference count by 1. 
  array->add_ref();
  return array;
}

Array *__quantum__rt__array_concatenate(Array *head, Array *tail) {
  if (head && tail) {
    auto resultArray = new Array(*head);
    resultArray->append(*tail);
    if (verbose)
      std::cout << "[qir-qrt] Concatenate two arrays of size " << head->size()
                << " and " << tail->size() << ".\n";
    return resultArray;
  }

  return nullptr;
}

// Creates a new array. The first int is the size of each element in bytes. The
// second int is the dimension count. The variable arguments should be a
// sequence of int64_ts contains the length of each dimension. The bytes of the
// new array should be set to zero.
Array *__quantum__rt__array_create_nonvariadic(int itemSizeInBytes,
                                               int countDimensions,
                                               va_list dims) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}

Array *__quantum__rt__array_create(int itemSizeInBytes, int countDimensions,
                                   ...) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}

int8_t *__quantum__rt__array_get_element_ptr_nonvariadic(Array *array,
                                                         va_list args) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}

// Returns a pointer to the indicated element of the array. The variable
// arguments should be a sequence of int64_ts that are the indices for each
// dimension.
int8_t *__quantum__rt__array_get_element_ptr(Array *array, ...) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}

// Creates and returns an array that is a projection of an existing array. The
// int indicates which dimension the projection is on, and the int64_t specifies
// the specific index value to project. The returned Array* will have one fewer
// dimension than the existing array.
Array *__quantum__rt__array_project(Array *array, int dim, int64_t index) {
  if (verbose)
    std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  return nullptr;
}
}