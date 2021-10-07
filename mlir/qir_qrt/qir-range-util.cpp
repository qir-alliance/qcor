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

namespace qcor {
std::vector<int64_t> getRangeValues(Array *in_array, const Range &in_range) {
  const bool is_fwd_range = in_range.step > 0;

  const auto convertIndex = [&](int64_t in_rawIdx) -> int64_t {
    if (in_rawIdx >= 0) {
      return in_rawIdx;
    }
    // Negative-based index:
    // in_rawIdx = -1 => size - 1 (last element)
    int64_t result = in_array->size() + in_rawIdx;
    if (result < 0) {
      throw std::invalid_argument("range");
    }
    return result;
  };

  // Convert to absolute index.
  const auto start_idx = convertIndex(in_range.start);
  const auto end_idx = convertIndex(in_range.end);
  // start == end
  if (start_idx == end_idx) {
    return {end_idx};
  }

  if (is_fwd_range) {
    if (start_idx > end_idx) {
      return {};
    } else {
      assert(in_range.step > 0);
      std::vector<int64_t> result;
      for (int64_t i = start_idx; i <= end_idx; i += in_range.step) {
        result.emplace_back(i);
      }
      return result;
    }
  } else {
    if (start_idx < end_idx) {
      return {};
    } else {
      std::vector<int64_t> result;
      assert(in_range.step < 0);
      for (int64_t i = start_idx; i >= end_idx; i += in_range.step) {
        result.emplace_back(i);
      }
      return result;
    }
  }
}
} // namespace qcor