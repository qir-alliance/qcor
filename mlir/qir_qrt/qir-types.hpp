#pragma once

#include <atomic>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <cstring>
// Defines implementations of QIR Opaque types

namespace qcor {
namespace internal {
// Internal tracker to make sure we're doing proper
// ref. counting, e.g. when generating QIR LLVM IR.
class AllocationTracker {
public:
  static AllocationTracker &get();

  void onAllocate(void *objPtr) {
    assert(m_refCountMap.find(objPtr) == m_refCountMap.end());
    m_refCountMap[objPtr] = 1;
  }

  void onDeallocate(void *objPtr) {
    assert(m_refCountMap.find(objPtr) != m_refCountMap.end());
    // Remove from the tracking map
    // (allocation by new may return the same address)
    m_refCountMap.erase(objPtr);
  }

  void updateCount(void *objPtr, int newCount) {
    assert(m_refCountMap.find(objPtr) != m_refCountMap.end());
    assert(newCount >= 0);
    m_refCountMap[objPtr] = newCount;
  }

  // Check if we have any leakage.
  // Returns false if no leak, true otherwise.
  // Can be use at shut-down (Finalize) to detect leakage.
  bool checkLeak() const {
    for (const auto &[ptr, count] : m_refCountMap) {
      if (count > 0) {
        return true;
      }
    }
    // No leak, all objects have been released.
    return false;
  }

private:
  AllocationTracker(){};
  static AllocationTracker *m_globalTracker;
  std::unordered_map<void *, int> m_refCountMap;
};
} // namespace internal
} // namespace qcor

// FIXME - Qubit should be a struct that keeps track of idx
// qreg name, array it comes from, and associated accelerator buffer
// Make this a struct now so that we can upgrade the API later
// more easily.
struct Qubit {
  const uint64_t id;
  operator int() const { return id; }
  // Allocation function:
  // Note: currently, we don't reclaim deallocated qubits.
  // until the very end of the quantum execution:
  // i.e. all qubit array are cleaned-up 
  // reset_counter() will be called.
  static Qubit *allocate() {
    Qubit *newQubit = new Qubit(q_counter);
    q_counter++;
    return newQubit;
  }

  // Create a QIR qubit with this ID
  // Rationale: should only be used for interoperability with non-QIR
  // environment.
  // i.e. actual qubits are allocated elsewhere, just need to create an alias
  // in QIR runtime to match up the qubit ID.
  static Qubit *create(uint64_t id) {
    Qubit *newQubit = new Qubit(id);
    return newQubit;
  }

  static void reset_counter() { q_counter = 0; }

private:
  Qubit(uint64_t idVal) : id(idVal) {}
  inline static uint64_t q_counter = 0;
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
        m_storage(nbItems * itemSizeInBytes, 0), m_refCount(1) {
    assert(m_itemSizeInBytes > 0);
    qcor::internal::AllocationTracker::get().onAllocate(this);
  };
  // Copy ctor:
  // note: we copy the Storage vector, hence set ref count to 1
  Array(const Array &other)
      : m_itemSizeInBytes(other.m_itemSizeInBytes), m_storage(other.m_storage),
        m_refCount(1) {
    qcor::internal::AllocationTracker::get().onAllocate(this);
  }

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
  void add_ref() {
    m_refCount += 1;
    qcor::internal::AllocationTracker::get().updateCount(this, m_refCount);
  }
  // Release a single ref.
  // Returns true if this Array should be deleted
  // if heap allocated (via new)
  bool release_ref() {
    m_refCount -= 1;
    qcor::internal::AllocationTracker::get().updateCount(this, m_refCount);
    return (m_refCount == 0);
  }

  int ref_count() const { return m_refCount; }

  ~Array() { qcor::internal::AllocationTracker::get().onDeallocate(this); }

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
  // Since the layout of this struct is important,
  // (we cast a byte array to this struct)
  // use a simple integer var as ref count.
  int32_t m_refCount;
  // Must be at the end
  int8_t m_data[];

  TuplePtr getTuple() { return m_data; }
  size_t tupleSize() const { return m_tupleSize; }
  static TupleHeader *create(int size) {
    int8_t *buffer = new int8_t[sizeof(TupleHeader) + size];
    TupleHeader *th = reinterpret_cast<TupleHeader *>(buffer);
    th->m_tupleSize = size;
    th->m_refCount = 1;
    qcor::internal::AllocationTracker::get().onAllocate(th);
    return th;
  }
  static TupleHeader *create(TupleHeader *other) {
    const auto size = other->m_tupleSize;
    int8_t *buffer = new int8_t[sizeof(TupleHeader) + size];
    TupleHeader *th = reinterpret_cast<TupleHeader *>(buffer);
    th->m_tupleSize = size;
    th->m_refCount = 1;
    memcpy(th->m_data, other->m_data, size);
    qcor::internal::AllocationTracker::get().onAllocate(th);
    return th;
  }

  static TupleHeader *getHeader(TuplePtr tuple) {
    return reinterpret_cast<TupleHeader *>(tuple -
                                           offsetof(TupleHeader, m_data));
  }

  // Ref. counting:
  void add_ref() {
    m_refCount += 1;
    qcor::internal::AllocationTracker::get().updateCount(this, m_refCount);
  }
  
  // Release a single ref. and delete the Tuple
  // if the ref. count == 0.
  // Note: since we use a heap-allocated byte array to represent a tuple-header,
  // hence we need to clean up here accordingly.
  // Returns true if the Tuple has been deleted.
  bool release_ref() {
    m_refCount -= 1;
    qcor::internal::AllocationTracker::get().updateCount(this, m_refCount);
    if (m_refCount == 0) {
      // Re-cast this to a byte-buffer.
      int8_t *buffer = reinterpret_cast<int8_t *>(this);
      qcor::internal::AllocationTracker::get().onDeallocate(this);
      delete[] buffer;
      return true;
    }

    return false;
  }

  int ref_count() const { return m_refCount; }
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
} // namespace qcor

// QIR Callable implementation.
struct Callable {
  // Typedef's and constants
  typedef void (*CallableEntryType)(TuplePtr, TuplePtr, TuplePtr);
  typedef void (*CaptureCallbackType)(TuplePtr, int32_t);
  static int constexpr AdjointIdx = 1;
  static int constexpr ControlledIdx = 1 << 1;
  static int constexpr TableSize = 4;
  static int constexpr CaptureCallbacksTableSize = 2;
  // =======================================================

  void invoke(TuplePtr args, TuplePtr result);
  // Constructor from C++ functor
  Callable(qcor::qsharp::IFunctor *in_functor) : m_functor(in_functor) {}
  Callable(CallableEntryType *ftEntries, CaptureCallbackType *captureCallbacks,
           TuplePtr capture) {
    memcpy(m_functionTable, ftEntries, sizeof(m_functionTable));
    if (m_functionTable[0] == nullptr) {
      throw "Base functor must be defined.";
    }
    if (captureCallbacks != nullptr) {
      memcpy(m_captureCallbacks, captureCallbacks,
             sizeof(this->m_captureCallbacks));
    }
    m_capture = capture;
  }

  // Add arbitrary nested layer of control/adjoint
  // A + A = I; A + C = C + A = CA; C + C = C; CA + A = C; CA + C = CA
  void applyFunctor(int functorIdx) {
    if (functorIdx == Callable::AdjointIdx) {
      m_functorIdx ^= Callable::AdjointIdx;
      if (m_functionTable[m_functorIdx] == nullptr) {
        printf("The Callable doesn't have Adjoint implementation.");
        throw;
      }
    }
    if (functorIdx == Callable::ControlledIdx) {
      m_functorIdx |= Callable::ControlledIdx;
      if (m_functionTable[m_functorIdx] == nullptr) {
        printf("The Callable doesn't have Controlled implementation.");
        throw;
      }
      m_controlledDepth++;
    }
  }

private:
  qcor::qsharp::IFunctor *m_functor = nullptr;
  CallableEntryType m_functionTable[TableSize] = {nullptr, nullptr, nullptr, nullptr};
  CaptureCallbackType m_captureCallbacks[CaptureCallbacksTableSize] = {nullptr, nullptr};
  TuplePtr m_capture = nullptr;
  int m_functorIdx = 0;
  int m_controlledDepth = 0;
};

// QIR string type (regular string with ref. counting)
struct QirString {
  int32_t m_refCount;
  std::string m_str;

  QirString(std::string &&str) : m_refCount(1), m_str(str) {}
  QirString(const char *cstr) : m_refCount(1), m_str(cstr) {}
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
} // namespace qsharp
} // namespace qcor
