#include <iostream>
#include <vector>

qcor_include_qsharp(QCOR__Testing__TestFunctors__Interop, int64_t);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qrt ftqc bell.qs bell_driver.cpp
// Run with:
// $ ./a.out
int main() {
  const auto error_code = QCOR__Testing__TestFunctors__Interop();
  std::cout << "Error code: " << error_code << "\n";
  qcor_expect(error_code == 0);
  return 0;
}