#include <memory>
#include <vector>
#include "qalloc.hpp"
#include "qir-types.hpp"

namespace xacc {
    class AcceleratorBuffer;
}

extern "C" {
using qreg = xacc::internal_compiler::qreg;
extern Result ResultZero;
extern Result ResultOne;
extern unsigned long allocated_qbits;
extern bool initialized;
extern bool verbose;
// Global register instance.
extern std::shared_ptr<xacc::AcceleratorBuffer> global_qreg;
extern QRT_MODE mode;

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
void __quantum__qis__reset(Qubit* q);

void __quantum__qis__x(Qubit* q);
void __quantum__qis__y(Qubit* q);
void __quantum__qis__z(Qubit* q);

void __quantum__qis__rx(double x, Qubit* q);
void __quantum__qis__ry(double x, Qubit* q);
void __quantum__qis__rz(double x, Qubit* q);
void __quantum__qis__u3(double theta, double phi, double lambda, Qubit* q);

// New QIR runtime API:
void __quantum__qis__exp__body(Array* paulis, double angle, Array* qubits);
void __quantum__qis__exp__adj(Array* paulis, double angle, Array* qubits);
void __quantum__qis__exp__ctl(Array* ctls, Array* paulis, double angle, Array* qubits);
void __quantum__qis__exp__ctladj(Array* ctls, Array* paulis, double angle, Array* qubits);
void __quantum__qis__h__body(Qubit* q);
void __quantum__qis__h__ctl(Array* ctls, Qubit* q);
void __quantum__qis__r__body(Pauli pauli, double theta, Qubit* q);
void __quantum__qis__r__adj(Pauli pauli, double theta, Qubit* q);    
void __quantum__qis__r__ctl(Array* ctls, Pauli pauli, double theta, Qubit* q);
void __quantum__qis__r__ctladj(Array* ctls, Pauli pauli, double theta, Qubit* q);
void __quantum__qis__s__body(Qubit* q);
void __quantum__qis__s__adj(Qubit* q);
void __quantum__qis__s__ctl(Array* ctls, Qubit* q);
void __quantum__qis__s__ctladj(Array* ctls, Qubit* q);
void __quantum__qis__t__body(Qubit* q);
void __quantum__qis__t__adj(Qubit* q);
void __quantum__qis__t__ctl(Array* ctls, Qubit* q);
void __quantum__qis__t__ctladj(Array* ctls, Qubit* q);
void __quantum__qis__x__body(Qubit* q);
void __quantum__qis__x__adj(Qubit* q);
void __quantum__qis__x__ctl(Array* ctls, Qubit* q);
void __quantum__qis__x__ctladj(Array* ctls, Qubit* q);
void __quantum__qis__y__body(Qubit* q);
void __quantum__qis__y__adj(Qubit* q);
void __quantum__qis__y__ctl(Array* ctls, Qubit* q);
void __quantum__qis__y__ctladj(Array* ctls, Qubit* q);
void __quantum__qis__z__body(Qubit* q);
void __quantum__qis__z__adj(Qubit* q);
void __quantum__qis__z__ctl(Array* ctls, Qubit* q);
void __quantum__qis__z__ctladj(Array* ctls, Qubit* q);

void __quantum__qis__rx__body(double theta, Qubit* q);
void __quantum__qis__ry__body(double theta, Qubit* q);
void __quantum__qis__rz__body(double theta, Qubit* q);
void __quantum__qis__cnot__body(Qubit* src, Qubit* tgt);

Result* __quantum__qis__measure__body(Array* bases, Array* qubits);
double __quantum__qis__intasdouble__body(int32_t intVal);
void __quantum__rt__array_update_alias_count(Array* bases, int64_t count);
bool __quantum__rt__result_equal(Result* res, Result* comp);
int64_t __quantum__rt__array_get_size_1d(Array* state1);
int8_t* __quantum__rt__tuple_create(int64_t state);
void __quantum__rt__string_update_reference_count(void* str, int64_t count);
void __quantum__rt__array_update_reference_count(Array* aux, int64_t count);
void __quantum__rt__result_update_reference_count(Result *, int64_t count);
// =====================================================

Result* __quantum__qis__mz(Qubit* q);

// Qubit Alloc/Dealloc API
Array* __quantum__rt__qubit_allocate_array(uint64_t idx);
void __quantum__rt__qubit_release_array(Array* q);

// Array API
// Create an array
Array *__quantum__rt__array_create_1d(int32_t itemSizeInBytes,
                                      int64_t count_items);
// Get size
int64_t __quantum__rt__array_get_size_1d(Array *array);
// Get element at an index
int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t idx);
// Copy
Array *__quantum__rt__array_copy(Array *array, bool forceNewInstance);
// Concatenate
Array *__quantum__rt__array_concatenate(Array *head, Array *tail);
// Ref. counting
void __quantum__rt__array_update_alias_count(Array *array, int64_t increment);
void __quantum__rt__array_update_reference_count(Array *aux, int64_t count);

// Multi-dimension Array API
int32_t __quantum__rt__array_get_dim(Array *array);
int64_t __quantum__rt__array_get_size(Array *array, int32_t dim);
Array *__quantum__rt__array_create_nonvariadic(int itemSizeInBytes,
                                               int countDimensions,
                                               va_list dims);
Array *__quantum__rt__array_create(int itemSizeInBytes, int countDimensions,
                                   ...);
int8_t *__quantum__rt__array_get_element_ptr_nonvariadic(Array *array,
                                                         va_list args);
int8_t *__quantum__rt__array_get_element_ptr(Array *array, ...);
Array *__quantum__rt__array_project(Array *array, int dim, int64_t index);
}