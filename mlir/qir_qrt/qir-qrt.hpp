#include <memory>
#include <vector>
#include "qalloc.hpp"

#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wsign-compare"

extern "C" {

// FIXME - Qubit should be a struct that keeps track of idx
// qreg name, array it comes from, and associated accelerator buffer

using Qubit = uint64_t;
using Result = uint64_t;
using Array = std::vector<int8_t*>;
using TupleHeader = int *;
using qreg = xacc::internal_compiler::qreg;

extern Result ResultZero;
extern Result ResultOne;
extern unsigned long allocated_qbits;
extern bool initialized;

void initialize();
void __quantum__rt__initialize(int argc, int8_t** argv);
void __quantum__rt__finalize();

void __quantum__rt__set_external_qreg(qreg* q);

void __quantum__qis__cnot(Qubit* src, Qubit* tgt);
void __quantum__qis__h(Qubit* q);
void __quantum__qis__s(Qubit* q);
void __quantum__qis__sdg(Qubit* q);
void __quantum__qis__t(Qubit* q);
void __quantum__qis__tdg(Qubit* q);

void __quantum__qis__x(Qubit* q);
void __quantum__qis__y(Qubit* q);
void __quantum__qis__z(Qubit* q);

void __quantum__qis__rx(double x, Qubit* q);
void __quantum__qis__ry(double x, Qubit* q);
void __quantum__qis__rz(double x, Qubit* q);

Result* __quantum__qis__mz(Qubit* q);

Array* __quantum__rt__qubit_allocate_array(uint64_t idx);

// Array functions
int8_t * __quantum__rt__array_get_element_ptr_1d(Array* q,
                                                      uint64_t idx);
void __quantum__rt__qubit_release_array(Array* q);

}