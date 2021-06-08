#include <iostream>
#include <vector>

qcor_include_qsharp(QCOR__Testing__TestFunctors__Interop, void);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor -qrt ftqc bell.qs bell_driver.cpp
// Run with:
// $ ./a.out
int main() {
  QCOR__Testing__TestFunctors__Interop();
  return 0;
}