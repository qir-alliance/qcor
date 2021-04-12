#include <iostream> 
#include <vector>
#include "qir-types-utils.hpp"

// Include the external QSharp function.
qcor_include_qsharp(QCOR__TestKernel__body, ::Array*);
qcor_include_qsharp(QCOR__TestClean__body, void);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qrt ftqc kernel.qs driver.cpp
// Run with:
// $ ./a.out
int main() {
  // Kernel that clean-up all allocated objects.
  QCOR__TestClean__body();
  // No leak expected.
  assert(!qcor::internal::AllocationTracker::get().checkLeak());
  // This kernel returns an Array,
  // (allocated in the kernel body)
  auto test = QCOR__TestKernel__body();
  const auto resultVec = qcor::qir::fromArray<double>(test);
  // Should detect a leak.
  assert(qcor::internal::AllocationTracker::get().checkLeak());

  for (const auto &el : resultVec) {
    std::cout << el << "\n";
  }

  // Release the ref-count of the returned array.
  // This should dealloc the Array
  test->release_ref();
  // No leak expected.
  assert(!qcor::internal::AllocationTracker::get().checkLeak());

  return 0;
}