#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Quantum dialect definition.
//===----------------------------------------------------------------------===//

def QuantumDialect : Dialect {
    let name = "quantum";
    let summary = "A MLIR dialect for quantum assembly languages.";
    let description = [{
        This dialect provides operations specific to quantum assembly languages.
    }];
    let cppNamespace = "::mlir::quantum";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class QuantumOp<string mnemonic, list<OpTrait> traits = []> :
        Op<QuantumDialect, mnemonic, traits>;

#endif // QUANTUM_DIALECT