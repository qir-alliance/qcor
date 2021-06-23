
#include "Quantum/QuantumDialect.h"
#include "Quantum/QuantumOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::quantum;
namespace {
/// Inliner interface
/// This class defines the interface for handling inlining with Quantum
/// operations.
// We simplify inherit from the base interface class and override
/// the necessary methods.
struct QuantumInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// This hook checks to see if the given callable operation is legal to inline
  /// into the given call. 
  /// Operations in Quantum dialect are always legal to inline.
  bool isLegalToInline(Operation *, Operation *, bool) const final {
    return true;
  }

  /// This hook checks to see if the given operation is legal to inline into the
  /// given region.
  /// Always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // namespace
//===----------------------------------------------------------------------===//
// Quantum dialect.
//===----------------------------------------------------------------------===//

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quantum/QuantumOps.cpp.inc"
      >();
  addInterfaces<QuantumInlinerInterface>();
}

// static void print(mlir::OpAsmPrinter &printer, mlir::quantum::InstOp op) {
//   printer << "q." << op.name() << "(" << *(op.qubits().begin());
//   for (auto i = 1; i < op.qubits().size(); i++) {
//     printer << ", " << op.qubits()[i];
//   }
//   printer << ")";

// }