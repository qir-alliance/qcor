#pragma once

#include <memory>
#include <stack>
#include <vector>

#include "qalloc.hpp"
#include "qir-types.hpp"

namespace xacc {
class AcceleratorBuffer;
}

extern "C" {
using qreg = xacc::internal_compiler::qreg;

// Note: The QIR spec requires ResultZero and ResultOne
// as External Global constants of *pointer* type.
extern Result *ResultZero;
extern Result *ResultOne;
extern unsigned long allocated_qbits;
extern bool initialized;
extern bool verbose;
extern std::string qpu_name;
// Global register instance.
extern std::shared_ptr<xacc::AcceleratorBuffer> global_qreg;
extern QRT_MODE mode;

void initialize();

// Initialize/Finalize/Config API
// __attribute__((constructor)) void __init_qir_qrt(int argc, char** argv);
void __quantum__rt__initialize(int argc, int8_t **argv);
void __quantum__rt__finalize();
void __quantum__rt__set_external_qreg(qreg *q);

// QIS API (i.e. quantum instructions)
void __quantum__qis__swap(Qubit *src, Qubit *tgt);
void __quantum__qis__cnot(Qubit *src, Qubit *tgt);
void __quantum__qis__cphase(double x, Qubit *src, Qubit *tgt);
void __quantum__qis__h(Qubit *q);
void __quantum__qis__s(Qubit *q);
void __quantum__qis__sdg(Qubit *q);
void __quantum__qis__t(Qubit *q);
void __quantum__qis__tdg(Qubit *q);
void __quantum__qis__reset(Qubit *q);

void __quantum__qis__x(Qubit *q);
void __quantum__qis__y(Qubit *q);
void __quantum__qis__z(Qubit *q);

void __quantum__qis__rx(double x, Qubit *q);
void __quantum__qis__ry(double x, Qubit *q);
void __quantum__qis__rz(double x, Qubit *q);
void __quantum__qis__u3(double theta, double phi, double lambda, Qubit *q);
Result *__quantum__qis__mz(Qubit *q);
// Compare results.
bool __quantum__rt__result_equal(Result *res, Result *comp);
void __quantum__rt__result_update_reference_count(Result *, int32_t count);
// Get reference Result.
Result *__quantum__rt__result_get_one();
Result *__quantum__rt__result_get_zero();

// Qubit Alloc/Dealloc API
Array *__quantum__rt__qubit_allocate_array(uint64_t idx);
void __quantum__rt__qubit_release_array(Array *q);

void __quantum__rt__start_ctrl_u_region();
void __quantum__rt__end_ctrl_u_region(Qubit *ctrl_qubit);
void __quantum__rt__start_adj_u_region();
void __quantum__rt__end_adj_u_region();
void __quantum__rt__start_pow_u_region();
void __quantum__rt__end_pow_u_region(int64_t power);

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
// Slice
// Creates and returns an array that is a slice of an existing array.
// The int dim which dimension the slice is on (0 for 1d arrays).
// The Range range specifies the slice.
// Note: QIR defines a Range as type { i64, i64, i64 }
// i.e. a struct of 3 int64_t
// and define an API at the *LLVM IR* level of passing this by value
// i.e. the signature is %Range, not "%struct.Range* byval(%struct.Range)"
// Hence, it is actually equivalent to an expanded list of struct member.
// https://lists.llvm.org/pipermail/llvm-dev/2018-April/122714.html
// Until the spec. is updated (see
// https://github.com/microsoft/qsharp-language/issues/31) this is actually the
// C-ABI that will match the QIR IR.
Array *__quantum__rt__array_slice(Array *array, int32_t dim,
                                  int64_t range_start, int64_t range_step,
                                  int64_t range_end);
// Note: Overloading is not possible in C, so just keep the implementation in
// this local func.
Array *quantum__rt__array_slice(Array *array, int32_t dim, Range range);

// 1D-Array slice
Array *__quantum__rt__array_slice_1d(Array *array, int64_t range_start,
                                     int64_t range_step, int64_t range_end);
// Ref. counting
void __quantum__rt__array_update_alias_count(Array *array, int32_t increment);
void __quantum__rt__array_update_reference_count(Array *aux, int32_t count);

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

// String-related API:
// Specs: https://github.com/microsoft/qsharp-language/blob/main/Specifications/QIR/Data-Types.md#strings
void __quantum__rt__string_update_reference_count(QirString *str,
                                                  int32_t count);
QirString *__quantum__rt__string_create(char *null_terminated_buffer);
QirString *__quantum__rt__string_concatenate(QirString *in_head,
                                             QirString *in_tail);
bool __quantum__rt__string_equal(QirString *lhs, QirString *rhs);
QirString *__quantum__rt__int_to_string(int64_t val);
QirString *__quantum__rt__double_to_string(double val);
QirString *__quantum__rt__bool_to_string(bool val);
QirString *__quantum__rt__result_to_string(Result *val);
QirString *__quantum__rt__pauli_to_string(Pauli pauli);
QirString *__quantum__rt__qubit_to_string(Qubit *q);
QirString *__quantum__rt__range_to_string(int64_t range_start,
                                          int64_t range_step,
                                          int64_t range_end);
const char *__quantum__rt__string_get_data(QirString *str);
int32_t __quantum__rt__string_get_length(QirString *str);

// Tuples:
TuplePtr __quantum__rt__tuple_create(int64_t size);
void __quantum__rt__tuple_update_reference_count(TuplePtr th, int32_t c);
void __quantum__rt__tuple_update_alias_count(TuplePtr th, int32_t c);

// Callables:
void __quantum__rt__callable_update_reference_count(Callable *clb, int32_t c);
void __quantum__rt__callable_update_alias_count(Callable *clb, int32_t c);
void __quantum__rt__callable_invoke(Callable *clb, TuplePtr args, TuplePtr res);
Callable *__quantum__rt__callable_copy(Callable *clb, bool force);
void __quantum__rt__capture_update_reference_count(Callable *clb,
                                                   int32_t count);
void __quantum__rt__capture_update_alias_count(Callable *clb, int32_t count);
void __quantum__rt__callable_memory_management(int32_t index, Callable *clb,
                                               int64_t parameter);
Callable *__quantum__rt__callable_make_adjoint(Callable *clb);
Callable *__quantum__rt__callable_make_controlled(Callable *clb);
// Implementation table: 4x callables of a specific signature
typedef struct impl_table_t {
  void (*f[4])(TuplePtr, TuplePtr, TuplePtr);
} impl_table_t;
typedef struct mem_management_cb_t {
  void (*f[2])(TuplePtr, int64_t);
} mem_management_cb_t;
// Create callable (from Q#): 
// See spec: https://github.com/microsoft/qsharp-language/blob/main/Specifications/QIR/Callables.md
Callable* __quantum__rt__callable_create(impl_table_t* ft, mem_management_cb_t* callbacks, TuplePtr capture);
// Classical Runtime: 
// https://github.com/microsoft/qsharp-language/blob/main/Specifications/QIR/Classical-Runtime.md#classical-runtime
void __quantum__rt__fail(QirString *str);
void __quantum__rt__message(QirString *str);
}

namespace qcor {
using qubit = Qubit *;
struct qreg {
 protected:
  Array *data;
  uint64_t m_size;

 public:
  qreg(const qreg &other) {
    data = other.data;
    m_size = other.m_size;
  }
  qreg(const uint64_t size) {
    data = __quantum__rt__qubit_allocate_array(size);
    m_size = size;
  }
  qubit operator[](const uint64_t element) {
    assert(element < m_size);
    auto ptr = __quantum__rt__array_get_element_ptr_1d(data, element);
    return reinterpret_cast<Qubit **>(ptr)[0];
  }
  Array* raw_array() {
      return data;
  }
  ~qreg() {
    __quantum__rt__qubit_release_array(data);
  }
};

void initialize();
void initialize(std::vector<std::string> args);
void initialize(int argc, char **argv);
qreg qalloc(const uint64_t size);
}  // namespace qcor
