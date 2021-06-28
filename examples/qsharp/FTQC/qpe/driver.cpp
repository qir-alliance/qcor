#include <iostream> 
#include <vector>


qcor_include_qsharp(QCOR__QuantumPhaseEstimation__Interop, int64_t)

int main() {
  auto result = QCOR__QuantumPhaseEstimation__Interop();
  std::cout << "Result decimal: " << result << "\n";
  qcor_expect(result == 4);
  return 0;
}