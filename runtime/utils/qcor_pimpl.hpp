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