#include <qcor_arithmetic>


// Compile:
int main(int argc, char **argv) {
  auto q = qalloc(7);
  // New implementation
  controlled_phi_add::print_kernel(q.head(5), 11, {q[5], q[6]});
  
  
  // Deprecated....
  std::cout << "Original:\n";
  ccPhiAdd::print_kernel(q, 5, 6, 11, 0, 5, 0);

  return 0;
}