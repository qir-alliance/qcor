#include "qir-qrt.hpp"

#include <alloca.h>

#include "qcor.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

Result ResultZero = 0;
Result ResultOne = 1;
unsigned long allocated_qbits = 0;
std::shared_ptr<xacc::AcceleratorBuffer> qbits;
std::shared_ptr<xacc::Accelerator> qpu;

// std::vector<unsigned long *> arrays;
std::map<unsigned long*, std::size_t> array_sizes;

std::vector<std::unique_ptr<Array>> allocated_arrays;

bool initialized = false;
void __quantum__rt__initialize(int argc, char** argv);

void initialize() {
  if (!initialized) {
    printf("[qsharp-qcor-adapter] Initializing FTQC runtime...\n");
    // qcor::set_verbose(true);
    xacc::internal_compiler::__qrt_env = "ftqc";
    xacc::Initialize();
    auto qpu = xacc::getAccelerator("aer");
    xacc::internal_compiler::qpu = qpu;
    ::quantum::qrt_impl = xacc::getService<::quantum::QuantumRuntime>(
        xacc::internal_compiler::__qrt_env);
    ::quantum::qrt_impl->initialize("empty");
    initialized = true;
  }
}

void __quantum__qis__cnot(Qubit* src, Qubit* tgt) {
  initialize();
  // printf("[qsharp-qcor-adapter] Applying CNOT %lu %lu \n", src->second,
  // tgt->second);
  std::size_t src_copy = *src;  // src->second;
  std::size_t tgt_copy = *tgt;  // tgt->second;
  ::quantum::cnot({"q", src_copy}, {"q", tgt_copy});
}

void __quantum__qis__h(Qubit* q) {
  initialize();
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);  // q->second;

  printf("[qsharp-qcor-adapter] Applying H %lu\n", qcopy);
  ::quantum::h({"q", qcopy});
}

// void __quantum__qis__s(Qubit* q) {
//   initialize();
//   printf("[qsharp-qcor-adapter] Applying S %lu\n", q);
//   ::quantum::s({"q", q});
// }

// void __quantum__qis__x(Qubit* q) {
//   initialize();
//   printf("[qsharp-qcor-adapter] Applying X %lu\n", q);
//   ::quantum::x({"q", q});
// }
// void __quantum__qis__z(Qubit* q) {
//   initialize();
//   printf("[qsharp-qcor-adapter] Applying Z %lu\n", q);
//   ::quantum::z({"q", q});
// }

// void __quantum__qis__rx(double x, Qubit* q) {
//   initialize();
//   printf("[qsharp-qcor-adapter] Applying Rx(%f) %lu\n", x, q);
//   ::quantum::rx({"q", q}, x);
// }

// void __quantum__qis__rz(double x, Qubit* q) {
//   initialize();
//   printf("[qsharp-qcor-adapter] Applying Rz(%f) %lu\n", x, q);
//   ::quantum::rz({"q", q}, x);
// }

Result* __quantum__qis__mz(Qubit* q) {
  initialize();
  printf("[qsharp-qcor-adapter] Measuring qubit %lu\n", q);
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);

  if (!qbits) {
    qbits = std::make_shared<xacc::AcceleratorBuffer>(allocated_qbits);
  }

  ::quantum::set_current_buffer(qbits.get());
  auto bit = ::quantum::mz({"q", qcopy});
  printf("[qsharp-qcor-adapter] Result was %d.\n", bit);
  return bit ? &ResultOne : &ResultZero;
}

Array* __quantum__rt__qubit_allocate_array(uint64_t size) {
  initialize();
  printf("[qsharp-qcor-adapter] Allocating qubit array of size %lu.\n", size);

  auto new_array = std::make_unique<Array>(size);
  for (uint64_t i = 0; i < size; i++) {
    printf("Creating bit %lu\n", i);
    auto qubit = new uint64_t;  // Qubit("q", i);
    *qubit = i;
    (*new_array)[i] = reinterpret_cast<int8_t*>(qubit);
    printf("Created and added\n");
  }

  allocated_qbits = size;
  if (!qbits) {
    qbits = std::make_shared<xacc::AcceleratorBuffer>(allocated_qbits);
    ::quantum::set_current_buffer(qbits.get());
  }

  printf("Made it here\n");

  auto raw_ptr = new_array.get();
  allocated_arrays.push_back(std::move(new_array));
  return raw_ptr;
}

int8_t* __quantum__rt__array_get_element_ptr_1d(Array* q, uint64_t idx) {
  Array& arr = *q;
  int8_t* ptr = arr[idx];
  Qubit* qq = reinterpret_cast<Qubit*>(ptr);

  printf("[qsharp-qcor-adapter] Returning qubit array element %lu, idx=%lu.\n",
         *qq, idx);
  return ptr;
}
