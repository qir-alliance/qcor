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
#pragma once

#include <memory>

namespace qcor {
template <typename T>
class qcor_pimpl {
 private:
  std::unique_ptr<T> m;

 public:
  qcor_pimpl();
  qcor_pimpl(const qcor_pimpl<T>&);
  template <typename... Args>
  qcor_pimpl(Args&&...);
  ~qcor_pimpl();
  T* operator->();
  T* operator->() const;
  T& operator*();
};
}