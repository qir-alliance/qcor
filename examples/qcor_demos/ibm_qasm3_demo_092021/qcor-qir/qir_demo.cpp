#define __INTERNAL__DISABLE__QCOR__QIR__COMPAT__ 1
#include "qir-qrt.hpp"

// QPE Problem
// In this example, we demonstrate a simple QPE algorithm, i.e.
// i.e. Oracle(|State>) = exp(i*Phase)*|State>
// and we need to estimate that Phase value.
// The Oracle in this case is a T gate and the eigenstate is |1>
// i.e. T|1> = exp(i*pi/4)|1>
// We use 3 counting bits => totally 4 qubits.

Qubit* extract_qubit(Array* a, int idx);
void iqft(Array* q);

void oracle(Qubit* q) { __quantum__qis__t(q); }

void qpe(Array* qreg) {
  auto input_size = __quantum__rt__array_get_size_1d(qreg);

  // Extract the counting qubits and the state qubit
  Array* counting_qubits =
      __quantum__rt__array_slice_1d(qreg, 0, 1, input_size - 2);
  auto n_counting = __quantum__rt__array_get_size_1d(counting_qubits);
  auto state_qubit = extract_qubit(qreg, input_size - 1);

  // Put it in |1> eigenstate
  __quantum__qis__x(state_qubit);

  for (int i = 0; i < n_counting; i++) {
    auto tmp_qubit = extract_qubit(counting_qubits, i);
    __quantum__qis__h(tmp_qubit);
  }

  // run ctr-oracle operations
  for (auto i : range(n_counting)) {
    const int nbCalls = 1 << i;
    for (auto j : range(nbCalls)) {
      auto tmp_ptr = __quantum__rt__array_get_element_ptr_1d(qreg, i);
      Qubit* tmp_qubit = reinterpret_cast<Qubit*>(tmp_ptr);
      auto ctrl_qubit = extract_qubit(counting_qubits, i);
      __quantum__rt__start_ctrl_u_region();
      oracle(state_qubit);
      __quantum__rt__end_ctrl_u_region(ctrl_qubit);
    }
  }

  // Run Inverse QFT on counting qubits
  iqft(counting_qubits);

  // Measure the counting qubits
  for (int i = 0; i < n_counting; i++) {
    __quantum__qis__mz(extract_qubit(counting_qubits, i));
  }
}

// Oracle I want to consider

int main(int argc, char** argv) {
  __quantum__rt__initialize(argc, reinterpret_cast<int8_t**>(argv));

  Array* qubits = __quantum__rt__qubit_allocate_array(4);
  qpe(qubits);
  __quantum__rt__qubit_release_array(qubits);

  __quantum__rt__finalize();
}

Qubit* extract_qubit(Array* a, int idx) {
  auto q_raw_ptr = __quantum__rt__array_get_element_ptr_1d(a, idx);
  return *reinterpret_cast<Qubit**>(q_raw_ptr);
}

void iqft(Array* q) {
  auto nbQubits = __quantum__rt__array_get_size_1d(q);

  // Swap qubits
  for (int qIdx = 0; qIdx < nbQubits / 2; ++qIdx) {
    auto first = extract_qubit(q, qIdx);
    auto second = extract_qubit(q, nbQubits - qIdx - 1);
    __quantum__qis__swap(first, second);
  }

  for (int qIdx = 0; qIdx < nbQubits - 1; ++qIdx) {
    auto tmp = extract_qubit(q, qIdx);
    __quantum__qis__h(tmp);
    int j = qIdx + 1;
    for (int y = qIdx; y >= 0; --y) {
      const double theta = -M_PI / std::pow(2.0, j - y);
      auto first = extract_qubit(q, j);
      auto second = extract_qubit(q, y);
      __quantum__qis__cphase(theta, first, second);
    }
  }

  auto last = extract_qubit(q, nbQubits - 1);
  __quantum__qis__h(last);
}