#include "qcor.hpp"

int main(int argc, char** argv) {
  int n = argc;
  double m = 22;  
  
  using namespace qcor;
 
  qpu_lambda<int, double> superposition(
      [](qreg q) {          // Provide the kernel lambda
        qpu_lambda_body({  // wrap function body in this macro
          print("n = ", n);
          print("m = ", m);
          for (int i = 0; i < n; i++) {
            H(q[0]);
          }
          Measure(q[0]);
        })
      },
      qpu_lambda_variables({"q"},
                           {"n", "m"}),  // Must provide variable names in order
      n, m);                             // Must provide the captured variables

  auto q = qalloc(1);
  superposition(q);
  q.print();
}