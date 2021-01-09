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
std::string qpu_name = "qpp";

std::vector<std::unique_ptr<Array>> allocated_arrays;

bool initialized = false;
void __quantum__rt__initialize(int argc, int8_t** argv) {
  
  char ** casted = reinterpret_cast<char**>(argv);
  std::vector<std::string> args(casted, casted+argc);

  for (auto [i, arg] : qcor::enumerate(args)) {
    if (arg == "-qpu") {
      qpu_name = args[i+1];
    }
  }

  initialize();
}

void initialize() {
  if (!initialized) {
    printf("[qir-qrt] Initializing FTQC runtime...\n");
    // qcor::set_verbose(true);
    xacc::internal_compiler::__qrt_env = "ftqc";
    xacc::Initialize();
    std::cout << "[qir-qrt] Running on " << qpu_name << " backend.\n";
    auto qpu = xacc::getAccelerator(qpu_name);
    xacc::internal_compiler::qpu = qpu;
    ::quantum::qrt_impl = xacc::getService<::quantum::QuantumRuntime>(
        xacc::internal_compiler::__qrt_env);
    ::quantum::qrt_impl->initialize("empty");
    initialized = true;
  }
}

void __quantum__qis__cnot(Qubit* src, Qubit* tgt) {
  std::size_t src_copy = reinterpret_cast<std::size_t>(src);
  std::size_t tgt_copy = reinterpret_cast<std::size_t>(tgt);
  printf("[qir-qrt] Applying CX %lu, %lu\n", src_copy, tgt_copy);
  ::quantum::cnot({"q", src_copy}, {"q", tgt_copy});
}

void __quantum__qis__h(Qubit* q) {
  std::size_t qcopy = reinterpret_cast<std::size_t>(q); 
  printf("[qir-qrt] Applying H %lu\n", qcopy);
  ::quantum::h({"q", qcopy});
}

// void __quantum__qis__s(Qubit* q) {
//   initialize();
//   printf("[qir-qrt] Applying S %lu\n", q);
//   ::quantum::s({"q", q});
// }

// void __quantum__qis__x(Qubit* q) {
//   initialize();
//   printf("[qir-qrt] Applying X %lu\n", q);
//   ::quantum::x({"q", q});
// }
// void __quantum__qis__z(Qubit* q) {
//   initialize();
//   printf("[qir-qrt] Applying Z %lu\n", q);
//   ::quantum::z({"q", q});
// }

// void __quantum__qis__rx(double x, Qubit* q) {
//   initialize();
//   printf("[qir-qrt] Applying Rx(%f) %lu\n", x, q);
//   ::quantum::rx({"q", q}, x);
// }

// void __quantum__qis__rz(double x, Qubit* q) {
//   initialize();
//   printf("[qir-qrt] Applying Rz(%f) %lu\n", x, q);
//   ::quantum::rz({"q", q}, x);
// }

Result* __quantum__qis__mz(Qubit* q) {
  initialize();
  printf("[qir-qrt] Measuring qubit %lu\n", reinterpret_cast<std::size_t>(q));
  std::size_t qcopy = reinterpret_cast<std::size_t>(q);

  if (!qbits) {
    qbits = std::make_shared<xacc::AcceleratorBuffer>(allocated_qbits);
  }

  ::quantum::set_current_buffer(qbits.get());
  auto bit = ::quantum::mz({"q", qcopy});
  printf("[qir-qrt] Result was %d.\n", bit);
  return bit ? &ResultOne : &ResultZero;
}

Array* __quantum__rt__qubit_allocate_array(uint64_t size) {
  initialize();
  printf("[qir-qrt] Allocating qubit array of size %lu.\n", size);

  auto new_array = std::make_unique<Array>(size);
  for (uint64_t i = 0; i < size; i++) {
    auto qubit = new uint64_t;  // Qubit("q", i);
    *qubit = i;
    (*new_array)[i] = reinterpret_cast<int8_t*>(qubit);
  }

  allocated_qbits = size;
  if (!qbits) {
    qbits = std::make_shared<xacc::AcceleratorBuffer>(allocated_qbits);
    ::quantum::set_current_buffer(qbits.get());
  }

  auto raw_ptr = new_array.get();
  allocated_arrays.push_back(std::move(new_array));
  return raw_ptr;
}

int8_t* __quantum__rt__array_get_element_ptr_1d(Array* q, uint64_t idx) {
  Array& arr = *q;
  int8_t* ptr = arr[idx];
  Qubit* qq = reinterpret_cast<Qubit*>(ptr);

  printf("[qir-qrt] Returning qubit array element %lu, idx=%lu.\n",
         *qq, idx);
  return ptr;
}

void __quantum__rt__qubit_release_array(Array* q) {
  for (std::size_t i = 0; i < allocated_arrays.size(); i++) {
    if (allocated_arrays[i].get() == q) {
      auto& array_ptr = allocated_arrays[i];
      auto array_size = array_ptr->size();
      printf("[qir-qrt] deallocating the qubit array of size %lu\n", array_size);
      for (int k = 0; k < array_size; k++) {
        delete (*array_ptr)[k];
      }
      array_ptr->clear();
    }
  }
}

void __quantum__rt__finalize() {
  std::cout << "[qir-qrt] Running finalization routine.\n";
}