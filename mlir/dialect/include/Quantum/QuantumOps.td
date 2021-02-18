#ifndef Quantum_OPS
#define Quantum_OPS

include "QuantumDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def QubitType : OpaqueType<"quantum", "Qubit", "opaque qubit type">;
def ResultType : OpaqueType<"quantum", "Result", "opaque result type">;
def ArrayType : OpaqueType<"quantum", "Array", "opaque array type">;
def ArgvType : OpaqueType<"quantum", "ArgvType", "opaque argv type">;
def QregType : OpaqueType<"quantum", "QregType", "opaque qreg type">;

def QallocOp : QuantumOp<"qalloc", []> {
    let arguments = (ins AnyI64Attr:$size, StrAttr:$name);
    let results = (outs ArrayType:$qubits);
}

def ResultAllocOp : QuantumOp<"ralloc", []> {
    let arguments = (ins AnyI64Attr:$size, StrAttr:$name);
    let results = (outs ArrayType:$creg);
}

def ExtractQubitOp : QuantumOp<"qextract", []> {
    let arguments = (ins ArrayType:$qreg, I64:$idx);
    let results = (outs QubitType:$qbit);
}

def ExtractCbitOp : QuantumOp<"cextract", []> {
    let arguments = (ins ArrayType:$creg, I64:$idx);
    let results = (outs ResultType:$cbit);
}

def InstOp : QuantumOp<"inst", [AttrSizedOperandSegments]> {
    let arguments = (ins StrAttr:$name, Variadic<QubitType>:$qubits, Variadic<F64>:$params);
    let results = (outs Optional<I64>:$bit);
}

def DeallocOp : QuantumOp<"dealloc", []> {
    let arguments = (ins ArrayType:$qubits);
    let results = (outs);
}

def QRTInitOp : QuantumOp<"init", []> {
    let arguments = (ins AnyI32:$argc, ArgvType:$argv);
    let results = (outs);
}

def QRTFinalizeOp : QuantumOp<"finalize", []> {
    let arguments = (ins);
    let results = (outs);
}

def SetQregOp : QuantumOp<"set_qreg", []> {
    let arguments = (ins QregType:$qreg);
    let results = (outs);
}

#endif // Quantum_OPS