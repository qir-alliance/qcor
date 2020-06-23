#include "arithmetic.hpp"

// Compile:
int main(int argc, char **argv) {
  // Allocate 22 qubits required for a 5-bit number (4n*2)
  auto q = qalloc(22);
  int a = 11;
  int N = 21;
  // Call entry-point kernel
  periodFinding(q, a, N);
 
  q.print();
  return 0;
}
