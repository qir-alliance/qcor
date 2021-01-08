#include <memory>
#include <vector>
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wsign-compare"

extern "C" {
    
using Qubit = uint64_t;
using Result = uint64_t;
using Array = std::vector<int8_t*>;
using TupleHeader = int *;

// using ArrayStorage = std::vector<std::shared_ptr<Array>>;

// extern ArrayStorage array_storage;
extern Result ResultZero;
extern Result ResultOne;
extern unsigned long allocated_qbits;
extern bool initialized;

void initialize();
void __quantum__rt__initialize(int argc, char** argv);

void __quantum__qis__cnot(Qubit* src, Qubit* tgt);
void __quantum__qis__h(Qubit* q);
// void __quantum__qis__s(Qubit* q);
// void __quantum__qis__x(Qubit* q);
// void __quantum__qis__z(Qubit* q);

// void __quantum__qis__rx(double x, Qubit* q);
// void __quantum__qis__rz(double x, Qubit* q);

Result* __quantum__qis__mz(Qubit* q);

Array* __quantum__rt__qubit_allocate_array(uint64_t idx);

// Array functions
int8_t * __quantum__rt__array_get_element_ptr_1d(Array* q,
                                                      uint64_t idx);
}