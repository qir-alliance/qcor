#include <iostream> 
#include <vector>
#include "qcor.hpp"

// Include the external QSharp function.
qcor_include_qsharp(XACC__Ansatz__body, void, Array*, Array*)


// Compile with:
// Include both the qsharp source and this driver file 
// in the command line.
// $ qcor -qrt ftqc bell.qs bell_driver.cpp
// Run with:
// $ ./a.out
int main() {
  int64_t bit0 = 0;
  int64_t bit1 = 1;
  int64_t bit2 = 2;
  std::vector<int64_t*> qReg { &bit0, &bit1, &bit2 };
  double theta0 = 1.0;
  double theta1 = 2.0;
  double theta2 = 3.0;
  std::vector<double*> thetas { &theta0, &theta1, &theta2 };

  XACC__Ansatz__body(&qReg, &thetas);
  std::cout << "HOWDY\n";
  return 0;
}