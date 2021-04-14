#include "qir-types.hpp"

namespace qcor {
namespace internal {

AllocationTracker *AllocationTracker::m_globalTracker = nullptr;

AllocationTracker &AllocationTracker::get() {
  if (!m_globalTracker) {
    m_globalTracker = new AllocationTracker();
  }
  return *m_globalTracker;
}
} // namespace internal
} // namespace qcor