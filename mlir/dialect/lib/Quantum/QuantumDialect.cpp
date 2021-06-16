
#include "Quantum/QuantumDialect.h"
#include "Quantum/QuantumOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::quantum;

//===----------------------------------------------------------------------===//
// Quantum dialect.
//===----------------------------------------------------------------------===//

void QuantumDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Quantum/QuantumOps.cpp.inc"
      >();
}

// static void print(mlir::OpAsmPrinter &printer, mlir::quantum::InstOp op) {
//   printer << "q." << op.name() << "(" << *(op.qubits().begin());
//   for (auto i = 1; i < op.qubits().size(); i++) {
//     printer << ", " << op.qubits()[i]; 
//   }
//   printer << ")";

// }