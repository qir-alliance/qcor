#pragma once

#include <cassert>
#include <stdexcept>
#include <atomic>
// Defines implementations of QIR Opaque types

// FIXME - Qubit should be a struct that keeps track of idx
// qreg name, array it comes from, and associated accelerator buffer
// Make this a struct now so that we can upgrade the API later
// more easily.
struct Qubit {
  const uint64_t id;
  operator int() const { return id; }
  // Allocation function:
  // Note: currently, we don't reclaim deallocated qubits.
  // TODO: track qubit deallocations for reuse...
  static Qubit *allocate() {
    static uint64_t counter = 0;
    Qubit *newQubit = new Qubit(counter);
    counter++;
    return newQubit;
  }

 private:
  Qubit(uint64_t idVal) : id(idVal) {}
};

using Result = bool;

struct Array {
  // Vector of bytes
  using Storage = std::vector<int8_t>;
  int8_t *getItemPointer(int64_t index) {
    assert(index >= 0);
    assert(static_cast<uint64_t>(index * m_itemSizeInBytes) < m_storage.size());
    return &m_storage.at(index * m_itemSizeInBytes);
  }
  int8_t *operator[](int64_t index) { return getItemPointer(index); }
  // Ctors
  // Default items are pointers.
  Array(int64_t nbItems, int itemSizeInBytes = sizeof(int8_t *))
      : m_itemSizeInBytes(itemSizeInBytes),
        // Initialized to zero
        m_storage(nbItems * itemSizeInBytes, 0),
        m_refCount(1) {
    assert(m_itemSizeInBytes > 0);
  };
  // Copy ctor:
  // note: we copy the Storage vector, hence set ref count to 1
  Array(const Array &other)
      : m_itemSizeInBytes(other.m_itemSizeInBytes), m_storage(other.m_storage),
        m_refCount(1) {}

  void append(const Array &other) {
    if (other.m_itemSizeInBytes != m_itemSizeInBytes) {
      throw std::runtime_error("Cannot append Arrays of different types.");
    }

    m_storage.insert(m_storage.end(), other.m_storage.begin(),
                     other.m_storage.end());
  }

  int64_t size() const { return m_storage.size() / m_itemSizeInBytes; }
  void clear() { m_storage.clear(); }
  int64_t element_size() const { return m_itemSizeInBytes; }

  // Ref. counting:
  void add_ref() { m_refCount += 1; }
  // Release a single ref. 
  // Returns true if this Array should be deleted
  // if heap allocated (via new) 
  bool release_ref() {
    m_refCount -= 1;
    return (m_refCount == 0);
  }

  int ref_count() const { return m_refCount; }

private:
  // Must be const, i.e. changing the element size is NOT allowed.
  const int m_itemSizeInBytes;
  Storage m_storage;
  std::atomic<int> m_refCount;
};

enum Pauli : int8_t {
  Pauli_I = 0,
  Pauli_X,
  Pauli_Z,
  Pauli_Y,
};

enum QRT_MODE { FTQC, NISQ };

// QIR Range type:
struct Range {
  int64_t start;
  int64_t step;
  int64_t end;
};

using TuplePtr = int8_t *;
struct TupleHeader {
  // Tuple data
  int32_t m_tupleSize;
  int8_t m_data[];

  TuplePtr getTuple() { return m_data; }
  size_t tupleSize() const { return m_tupleSize; }
  static TupleHeader *create(int size) {
    int8_t *buffer = new int8_t[sizeof(TupleHeader) + size];
    TupleHeader *th = reinterpret_cast<TupleHeader *>(buffer);
    th->m_tupleSize = size;
    return th;
  }
  static TupleHeader *getHeader(TuplePtr tuple) {
    return reinterpret_cast<TupleHeader *>(tuple -
                                           offsetof(TupleHeader, m_data));
  }
};

// Callable:
// Note: this implementation is not fully spec-conformed
// since we don't handle complex adj or ctrl callables.
// Currently, this is to support wrapping a C++ function (classical)
// and providing it to Q# for invocation.
// TODO: we need a strategy to incorporate MSFT runtime implementation
// to support callables that are *created* by Q# code.
// These Q#-created callables have many more features as described in:
// https://github.com/microsoft/qsharp-language/blob/main/Specifications/QIR/Callables.md
// Forward declare:
namespace qcor {
namespace qsharp {
class IFunctor;
}
}  // namespace qcor

// QIR Callable implementation.
struct Callable {
  void invoke(TuplePtr args, TuplePtr result);
  Callable(qcor::qsharp::IFunctor *in_functor) : m_functor(in_functor) {}

 private:
  qcor::qsharp::IFunctor *m_functor;
};

namespace qcor {
// Helper func.
std::vector<int64_t> getRangeValues(::Array *in_array, const ::Range &in_range);

namespace qsharp {
// A generic base class of qcor function-like objects
// that will be invoked by Q# as a callable.
class IFunctor {
 public:
  virtual void execute(TuplePtr args, TuplePtr result) = 0;
};
}  // namespace qsharp
}  // namespace qcor
