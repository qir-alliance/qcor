#include "arithmetic.hpp"

// Compile:
int main(int argc, char **argv) {
  // Allocate 18 qubits required for a 4-bit number (4n*2)
  auto q = qalloc(18);
  int a = 4;
  int N = 15;
  // Call entry-point kernel
  periodFinding(q, a, N);
 
  q.print();
  return 0;
}
