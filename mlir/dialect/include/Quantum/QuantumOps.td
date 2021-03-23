#ifndef Quantum_OPS
#define Quantum_OPS

include "QuantumDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def QubitType : OpaqueType<"quantum", "Qubit", "opaque qubit type">;
def ResultType : OpaqueType<"quantum", "Result", "opaque result type">;
def ArrayType : OpaqueType<"quantum", "Array", "opaque array type">;
def ArgvType : OpaqueType<"quantum", "ArgvType", "opaque argv type">;
def QregType : OpaqueType<"quantum", "QregType", "opaque qreg type">;
def StringType : OpaqueType<"quantum", "StringType", "opaque string type">;

def QallocOp : QuantumOp<"qalloc", []> {
    let arguments = (ins AnyI64Attr:$size, StrAttr:$name);
    let results = (outs ArrayType:$qubits);
}

// Create an array holding Qubit pointers for aliasing purposes,
// i.e. not allocating new qubits.
def QaliasArrayAllocOp : QuantumOp<"createQubitArray", []> {
    let arguments = (ins AnyI64Attr:$size, StrAttr:$name);
    let results = (outs ArrayType:$qubits);
}

def ExtractQubitOp : QuantumOp<"qextract", []> {
    let arguments = (ins ArrayType:$qreg, AnyInteger:$idx);
    let results = (outs QubitType:$qbit);
}

// Assign a qubit pointer (specified by the Qubit array and index) to an alias pointer. 
// Signature: void qassign(Array* destination_array, int destination_idx, Array* source_array, int source_idx)
def AssignQubitOp : QuantumOp<"qassign", []> {
    let arguments = (ins ArrayType:$dest_qreg, AnyInteger:$dest_idx, ArrayType:$src_qreg, AnyInteger:$src_idx);
    let results = (outs);
}

// Extract array slice
def ArraySliceOp : QuantumOp<"qarray_slice", []> {
    let arguments = (ins ArrayType:$qreg, Variadic<I64>:$slice_range);
    let results = (outs ArrayType:$array_slice);
}
// Array Concatenation
def ArrayConcatOp : QuantumOp<"qarray_concat", []> {
    let arguments = (ins ArrayType:$qreg1, ArrayType:$qreg2);
    let results = (outs ArrayType:$concat_array);
}

def StartCtrlURegion : QuantumOp<"start_ctrl_u_region", []> {
  let arguments = (ins);
  let results = (outs);
}

def EndCtrlURegion : QuantumOp<"end_ctrl_u_region", []> {
  let arguments = (ins QubitType:$ctrl_qubit);
  let results = (outs);
}

def StartAdjointURegion : QuantumOp<"start_adj_u_region", []> {
  let arguments = (ins);
  let results = (outs);
}

def EndAdjointURegion : QuantumOp<"end_adj_u_region", []> {
  let arguments = (ins);
  let results = (outs);
}

def StartPowURegion : QuantumOp<"start_pow_u_region", []> {
  let arguments = (ins);
  let results = (outs);
}

def EndPowURegion : QuantumOp<"end_pow_u_region", []> {
  let arguments = (ins AnyI64:$pow);
  let results = (outs);
}

def InstOp : QuantumOp<"inst", [AttrSizedOperandSegments]> {
    let arguments = (ins StrAttr:$name, Variadic<QubitType>:$qubits, Variadic<F64>:$params);
    let results = (outs Optional<ResultType>:$bit);
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

def PrintOp : QuantumOp<"print", []> {
    let arguments = (ins Variadic<AnyType>:$print_args);
    let results = (outs);
}

def CreateStringLiteralOp : QuantumOp<"createString", []> {
    let arguments = (ins StrAttr:$text, StrAttr:$varname);
    let results = (outs StringType:$result);
}

#endif // Quantum_OPS