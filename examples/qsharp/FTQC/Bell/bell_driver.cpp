#include <iostream> 
#include <vector>

// Include the external QSharp function.
// With EntryPoint() annotation, there are 3 functions generated:
// QCOR__TestBell__body(): raw Q# callable
// QCOR__TestBell(): EntryPoint type (with result printing), no return
// QCOR__TestBell__Interop(): InteropFriendly function (type casting to C-type function)
qcor_include_qsharp(QCOR__TestBell, void, int64_t)

// Compile with:
// Include both the qsharp source and this driver file 
// in the command line.
// -qs-build-exe to activate entry point generation.
// $ qcor -qrt ftqc -qs-build-exe bell.qs bell_driver.cpp
// Run with:
// $ ./a.out
int main() {
  QCOR__TestBell(1024);
  return 0;
}