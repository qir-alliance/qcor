#include <iostream> 
#include <vector>
#include "qcor.hpp"

// Include the external QSharp function.
qcor_include_qsharp(XACC__TestBell__body, int64_t, int64_t)


// Compile with:
// Include both the qsharp source and this driver file 
// in the command line.
// $ qcor -qrt ftqc bell.qs bell_driver.cpp
// Run with:
// $ ./a.out
int main() {
  auto oneCounts = XACC__TestBell__body(1024);
  std::cout << "Result = " << oneCounts << "\n";
  return 0;
}