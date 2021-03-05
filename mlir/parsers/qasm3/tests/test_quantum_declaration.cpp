#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkQuantumDeclaration) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit single_q;
qubit my_qreg[10];
qreg qq_qreg[25];
)#";
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);

  std::cout << "MLIR:\n" << mlir << "\n";

  auto expected = R"#(module  {
  func @main(%arg0: i32, %arg1: !quantum.ArgvType) -> i32 {
    "quantum.init"(%arg0, %arg1) : (i32, !quantum.ArgvType) -> ()
    call @__internal_mlir_test() : () -> ()
    "quantum.finalize"() : () -> ()
    %c0_i32 = constant 0 : i32
    return %c0_i32 : i32
  }
  func @__internal_mlir_test() {
    %0 = "quantum.qalloc"() {name = "single_q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "my_qreg", size = 10 : i64} : () -> !quantum.Array
    %3 = "quantum.qalloc"() {name = "qq_qreg", size = 25 : i64} : () -> !quantum.Array
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    "quantum.dealloc"(%3) : (!quantum.Array) -> ()
    return
  }
  func @test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected, mlir);
}

TEST(qasm3VisitorTester, checkClassicalDeclarations) {
  const std::string const_test = R"#(OPENQASM 3;
include "qelib1.inc";
const layers = 22;
const layers2 = layers / 2;
const t = layers * 3;
const d = 1.2;
const tt = d * 33.3;
const mypi = pi / 2;
)#";
  auto mlir = qcor::mlir_compile("qasm3", const_test, "const_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "Const Test MLIR:\n" << mlir << "\n";

  const std::string expected_const = R"#(module  {
  func @__internal_mlir_const_test() {
    %c22_i64 = constant 22 : i64
    %c2_i64 = constant 2 : i64
    %0 = divi_unsigned %c22_i64, %c2_i64 : i64
    %c3_i64 = constant 3 : i64
    %1 = muli %c22_i64, %c3_i64 : i64
    %cst = constant 1.200000e+00 : f64
    %cst_0 = constant 3.330000e+01 : f64
    %2 = mulf %cst, %cst_0 : f64
    %cst_1 = constant 3.1415926535897931 : f64
    %3 = "std.divf"(%cst_1, %c2_i64) : (f64, i64) -> f64
    return
  }
  func @const_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_const_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected_const, mlir);

  const std::string var_test = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 10;
float[32] f;
float[64] ff = 3.14;
bit result;
bit results[2];
creg c[22];
bool b;
bool bb = 1;
bool bbb = 0;
)#";
  auto var_mlir = qcor::mlir_compile("qasm3", var_test, "var_test",
                                     qcor::OutputType::MLIR, false);

  std::cout << "Var Test MLIR:\n" << var_mlir << "\n";
  const std::string expected_var = R"#(module  {
  func @__internal_mlir_var_test() {
    %c0_i32 = constant 0 : i32
    %c10_i32 = constant 10 : i32
    %c0_i64 = constant 0 : i64
    %0 = index_cast %c0_i64 : i64 to index
    %1 = alloca() : memref<1xi32>
    store %c10_i32, %1[%0] : memref<1xi32>
    %cst = constant 0.000000e+00 : f32
    %2 = index_cast %c0_i64 : i64 to index
    %3 = alloca() : memref<1xf32>
    store %cst, %3[%2] : memref<1xf32>
    %cst_0 = constant 3.140000e+00 : f64
    %4 = index_cast %c0_i64 : i64 to index
    %5 = alloca() : memref<1xf64>
    store %cst_0, %5[%4] : memref<1xf64>
    %6 = alloc() : memref<1xi1>
    %7 = alloc() : memref<2xi1>
    %8 = alloc() : memref<22xi1>
    %false = constant false
    %9 = index_cast %c0_i64 : i64 to index
    %10 = alloca() : memref<1xi1>
    store %false, %10[%9] : memref<1xi1>
    %true = constant true
    %11 = index_cast %c0_i64 : i64 to index
    %12 = load %true[%11] : i1
    %13 = alloca() : memref<1xi1>
    store %12, %13[%11] : memref<1xi1>
    %false_1 = constant false
    %14 = index_cast %c0_i64 : i64 to index
    %15 = load %false_1[%14] : i1
    %16 = alloca() : memref<1xi1>
    store %15, %16[%14] : memref<1xi1>
    return
  }
  func @var_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_var_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";

  EXPECT_EQ(expected_var, var_mlir);

  const std::string cast_op_test = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 1;
bool b = bool(i);

)#";
  auto cast_op_mlir = qcor::mlir_compile("qasm3", cast_op_test, "cast_op_test",
                                         qcor::OutputType::MLIR, false);

  std::cout << "Cast Test MLIR:\n" << cast_op_mlir << "\n";
  const std::string expected_cast = R"#(module  {
  func @__internal_mlir_cast_op_test() {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c0_i64 = constant 0 : i64
    %0 = index_cast %c0_i64 : i64 to index
    %1 = alloca() : memref<1xi32>
    store %c1_i32, %1[%0] : memref<1xi32>
    %2 = memref_cast %1 : memref<1xi32> to i1
    %3 = index_cast %c0_i64 : i64 to index
    %4 = load %2[%3] : i1
    %5 = alloca() : memref<1xi1>
    store %4, %5[%3] : memref<1xi1>
    return
  }
  func @cast_op_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_cast_op_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected_cast, cast_op_mlir);
}

TEST(qasm3VisitorTester, checkMeasurements) {
  const std::string measure_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
qubit qq[2];

bit r, s;
r = measure q;
s = measure qq[0];

bit rr[2];
rr[0] = measure qq[0];

bit xx[4];
qubit qqq[4];
xx = measure qqq;

bit y, yy[2];
measure q -> y;
measure qq -> yy;
measure qq[0] -> y;
)#";
  auto mlir = qcor::mlir_compile("qasm3", measure_test, "measure_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "measure_test MLIR:\n" << mlir << "\n";
  const std::string meas_test = R"#(module  {
  func @__internal_mlir_measure_test() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %3 = alloc() : memref<1xi1>
    %4 = alloc() : memref<1xi1>
    %5 = "quantum.inst"(%1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %6 = index_cast %c0_i64 : i64 to index
    store %5, %3[%6] : memref<1xi1>
    %7 = "quantum.qextract"(%2, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %8 = "quantum.inst"(%7) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %9 = index_cast %c0_i64 : i64 to index
    store %8, %4[%9] : memref<1xi1>
    %10 = alloc() : memref<2xi1>
    %11 = "quantum.inst"(%7) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %12 = index_cast %c0_i64 : i64 to index
    store %11, %10[%12] : memref<2xi1>
    %13 = alloc() : memref<4xi1>
    %14 = "quantum.qalloc"() {name = "qqq", size = 4 : i64} : () -> !quantum.Array
    %15 = index_cast %c0_i64 : i64 to index
    %16 = "quantum.qextract"(%14, %15) : (!quantum.Array, index) -> !quantum.Qubit
    %17 = "quantum.inst"(%16) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %17, %13[%15] : memref<4xi1>
    %c1_i64 = constant 1 : i64
    %18 = index_cast %c1_i64 : i64 to index
    %19 = "quantum.qextract"(%14, %18) : (!quantum.Array, index) -> !quantum.Qubit
    %20 = "quantum.inst"(%19) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %20, %13[%18] : memref<4xi1>
    %c2_i64 = constant 2 : i64
    %21 = index_cast %c2_i64 : i64 to index
    %22 = "quantum.qextract"(%14, %21) : (!quantum.Array, index) -> !quantum.Qubit
    %23 = "quantum.inst"(%22) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %23, %13[%21] : memref<4xi1>
    %c3_i64 = constant 3 : i64
    %24 = index_cast %c3_i64 : i64 to index
    %25 = "quantum.qextract"(%14, %24) : (!quantum.Array, index) -> !quantum.Qubit
    %26 = "quantum.inst"(%25) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %26, %13[%24] : memref<4xi1>
    %27 = alloc() : memref<1xi1>
    %28 = alloc() : memref<2xi1>
    %29 = "quantum.inst"(%1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %30 = index_cast %c0_i64 : i64 to index
    store %29, %27[%30] : memref<1xi1>
    %31 = index_cast %c0_i64 : i64 to index
    %32 = "quantum.qextract"(%2, %31) : (!quantum.Array, index) -> !quantum.Qubit
    %33 = "quantum.inst"(%32) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %33, %28[%31] : memref<2xi1>
    %34 = index_cast %c1_i64 : i64 to index
    %35 = "quantum.qextract"(%2, %34) : (!quantum.Array, index) -> !quantum.Qubit
    %36 = "quantum.inst"(%35) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %36, %28[%34] : memref<2xi1>
    %37 = "quantum.inst"(%7) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %38 = index_cast %c0_i64 : i64 to index
    store %37, %27[%38] : memref<1xi1>
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    "quantum.dealloc"(%14) : (!quantum.Array) -> ()
    return
  }
  func @measure_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_measure_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";

  EXPECT_EQ(meas_test, mlir);
}

TEST(qasm3VisitorTester, checkQuantumInsts) {
  const std::string qinst_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
h q;
ry(2.2) q;

qubit qq[2];
x qq[0];
CX qq[0], qq[1];
U(0.1,0.2,0.3) qq[1];
cx q, qq[1];

)#";
  auto mlir = qcor::mlir_compile("qasm3", qinst_test, "qinst_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "qinst_test MLIR:\n" << mlir << "\n";
  const std::string expected = R"#(module  {
  func @__internal_mlir_qinst_test() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.inst"(%1) {name = "h", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %cst = constant 2.200000e+00 : f64
    %3 = "quantum.inst"(%1, %cst) {name = "ry", operand_segment_sizes = dense<1> : vector<2xi32>} : (!quantum.Qubit, f64) -> none
    %4 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %5 = "quantum.qextract"(%4, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %6 = "quantum.inst"(%5) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %c1_i64 = constant 1 : i64
    %7 = "quantum.qextract"(%4, %c1_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %8 = "quantum.inst"(%5, %7) {name = "cnot", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!quantum.Qubit, !quantum.Qubit) -> none
    %cst_0 = constant 1.000000e-01 : f64
    %cst_1 = constant 2.000000e-01 : f64
    %cst_2 = constant 3.000000e-01 : f64
    %9 = "quantum.inst"(%7, %cst_0, %cst_1, %cst_2) {name = "u3", operand_segment_sizes = dense<[1, 3]> : vector<2xi32>} : (!quantum.Qubit, f64, f64, f64) -> none
    %10 = "quantum.inst"(%1, %7) {name = "cnot", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!quantum.Qubit, !quantum.Qubit) -> none
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%4) : (!quantum.Array) -> ()
    return
  }
  func @qinst_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_qinst_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";

  EXPECT_EQ(expected, mlir);
}

TEST(qasm3VisitorTester, checkSubroutine) {
  const std::string subroutine_test = R"#(OPENQASM 3;
include "qelib1.inc";
def xmeasure qubit:q -> bit { h q; return measure q; }
qubit q, qq[2];
bit r, rr[2];

rr[0] = xmeasure q;
r = xmeasure qq[0];
)#";
  auto mlir = qcor::mlir_compile("qasm3", subroutine_test, "subroutine_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "subroutine_test MLIR:\n" << mlir << "\n";

  const std::string s = R"#(module  {
  func @__internal_mlir_subroutine_test() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %3 = alloc() : memref<1xi1>
    %4 = alloc() : memref<2xi1>
    %5 = call @xmeasure(%1) : (!quantum.Qubit) -> i1
    store %5, %4[%c0_i64] : memref<2xi1>
    %6 = "quantum.qextract"(%2, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %7 = call @xmeasure(%6) : (!quantum.Qubit) -> i1
    store %7, %3[%c0_i64] : memref<1xi1>
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    return
  }
  func @subroutine_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_subroutine_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
  func @xmeasure(%arg0: !quantum.Qubit) -> i1 {
    %0 = "quantum.inst"(%arg0) {name = "h", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %1 = "quantum.inst"(%arg0) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    return %1 : i1
  }
}
)#";
  EXPECT_EQ(s, mlir);

  const std::string subroutine_test2 = R"#(OPENQASM 3;
include "qelib1.inc";
def xcheck qubit[4]:d, qubit:a -> bit {
  // reset a;
  for i in [0: 3] cx d[i], a;
  return measure a;
}
qubit q;
const n = 10;
def parity(bit[n]:cin) -> bit {
  bit c;
  for i in [0: n-1] {
    c ^= cin[i];
  }
  return c;
}
)#";
  auto mlir2 = qcor::mlir_compile("qasm3", subroutine_test2, "subroutine_test2",
                                  qcor::OutputType::MLIR, false);

  std::cout << "subroutine_test MLIR:\n" << mlir2 << "\n";

  const std::string expected2 = R"#(module  {
  func @__internal_mlir_subroutine_test2() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %c10_i64 = constant 10 : i64
  }
  func @subroutine_test2(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_subroutine_test2() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
  func @xcheck(%arg0: !quantum.Array, %arg1: !quantum.Qubit) -> i1 {
    %c0_i64 = constant 0 : i64
    %0 = index_cast %c0_i64 : i64 to index
    %1 = index_cast %c0_i64 : i64 to index
    %2 = alloca() : memref<1xi64>
    store %0, %2[%0] : memref<1xi64>
    %c3_i64 = constant 3 : i64
    %3 = index_cast %c3_i64 : i64 to index
    %c1_i64 = constant 1 : i64
    %4 = index_cast %c1_i64 : i64 to index
    %5 = load %2[%1] : memref<1xi64>
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %6 = load %2[%1] : memref<1xi64>
    %7 = cmpi "slt", %6, %3 : i64
    cond_br %7, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %8 = load %2[%1] : memref<1xi64>
    %9 = "quantum.qextract"(%arg0, %8) : (!quantum.Array, i64) -> !quantum.Qubit
    %10 = "quantum.inst"(%9, %arg1) {name = "cnot", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!quantum.Qubit, !quantum.Qubit) -> none
    br ^bb3
  ^bb3:  // pred: ^bb2
    %11 = load %2[%1] : memref<1xi64>
    %12 = "std.addi"(%11, %4) : (i64, index) -> i64
    store %12, %2[%1] : memref<1xi64>
    br ^bb1
  ^bb4:  // pred: ^bb1
    %13 = "quantum.inst"(%arg1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    return %13 : i1
  }
  func @parity(%arg0: memref<10xi1>) -> i1 {
    %0 = alloc() : memref<1xi1>
    %c0_i64 = constant 0 : i64
    %1 = index_cast %c0_i64 : i64 to index
    %2 = index_cast %c0_i64 : i64 to index
    %3 = alloca() : memref<1xi64>
    store %1, %3[%1] : memref<1xi64>
    %c9_i64 = constant 9 : i64
    %4 = index_cast %c9_i64 : i64 to index
    %c1_i64 = constant 1 : i64
    %5 = index_cast %c1_i64 : i64 to index
    %6 = load %3[%2] : memref<1xi64>
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %7 = load %3[%2] : memref<1xi64>
    %8 = cmpi "slt", %7, %4 : i64
    cond_br %8, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %9 = load %3[%2] : memref<1xi64>
    %10 = load %arg0[%9] : memref<10xi1>
    %11 = index_cast %c0_i64 : i64 to index
    %12 = load %0[%11] : memref<1xi1>
    %13 = xor %12, %10 : i1
    %14 = index_cast %c0_i64 : i64 to index
    store %13, %0[%14] : memref<1xi1>
    br ^bb3
  ^bb3:  // pred: ^bb2
    %15 = load %3[%2] : memref<1xi64>
    %16 = "std.addi"(%15, %5) : (i64, index) -> i64
    store %16, %3[%2] : memref<1xi64>
    br ^bb1
  ^bb4:  // pred: ^bb1
    %c0_i64_0 = constant 0 : i64
    %17 = index_cast %c0_i64_0 : i64 to index
    %18 = load %0[%17] : memref<1xi1>
    return %18 : i1
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    return
  }
}
)#";

  EXPECT_EQ(expected2, mlir2);
  
}

// TEST(qasm3VisitorTester, checkAddAssignment) {
//   const std::string add_assignment = R"#(OPENQASM 3;
// include "qelib1.inc";
// int[64] i = 1;
// i += 2;
// )#";
//   auto mlir = qcor::mlir_compile("qasm3", add_assignment, "add_assignment",
//                                  qcor::OutputType::MLIR, false);
//   std::cout << "add_assignment MLIR:\n" << mlir << "\n";

//   const std::string s = R"#(module  {
//   func @__internal_mlir_add_assignment() {
//     %c0_i64 = constant 0 : i64
//     %c1_i64 = constant 1 : i64
//     %0 = index_cast %c0_i64 : i64 to index
//     %1 = alloca() : memref<1xi64>
//     store %c1_i64, %1[%0] : memref<1xi64>
//     %c2_i64 = constant 2 : i64
//     %2 = index_cast %c0_i64 : i64 to index
//     %3 = load %1[%2] : memref<1xi64>
//     %4 = addi %3, %c2_i64 : i64
//     %5 = index_cast %c0_i64 : i64 to index
//     store %4, %1[%5] : memref<1xi64>
//     return
//   }
//   func @add_assignment(%arg0: !quantum.qreg) {
//     "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
//     call @__internal_mlir_add_assignment() : () -> ()
//     "quantum.finalize"() : () -> ()
//     return
//   }
// }
// )#";
//   EXPECT_EQ(s, mlir);
//   // auto mlir2 = qcor::mlir_compile("qasm3", add_assignment, "add_assignment",
//   //                                qcor::OutputType::LLVMMLIR, false);
//   // std::cout << "add_assignment MLIR:\n" << mlir2 << "\n";

//   //   auto mlir3 = qcor::mlir_compile("qasm3", add_assignment,
//   //   "add_assignment",
//   //                                qcor::OutputType::LLVMIR, false);
//   // std::cout << "add_assignment MLIR:\n" << mlir3 << "\n";
// }

TEST(qasm3VisitorTester, checkIfStmt) {
  const std::string if_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q, s, qq[2];
const layers = 2;

bit c;
c = measure q;

if (c == 1) {
    z s;
}

if (layers == 2) {
    z s;
} else {
    x s;
}

bit cc[2];
cc = measure qq;
if ( cc[1] == 1) {
  ry(2.2) s;
}

if ( 1 == cc[0] ) {
  ry(2.2) s;
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", if_stmt, "if_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << "if_stmt MLIR:\n" << mlir << "\n";
  const std::string test = R"#(module  {
  func @__internal_mlir_if_stmt() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "s", size = 1 : i64} : () -> !quantum.Array
    %3 = "quantum.qextract"(%2, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %4 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %c2_i64 = constant 2 : i64
    %5 = alloc() : memref<1xi1>
    %6 = "quantum.inst"(%1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %7 = index_cast %c0_i64 : i64 to index
    store %6, %5[%7] : memref<1xi1>
    %8 = index_cast %c0_i64 : i64 to index
    %9 = load %5[%8] : memref<1xi1>
    %c1_i64 = constant 1 : i64
    %10 = index_cast %c1_i64 : i64 to i1
    %11 = cmpi "eq", %9, %10 : i1
    cond_br %11, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %12 = "quantum.inst"(%3) {name = "z", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %13 = cmpi "eq", %c2_i64, %c2_i64 : i64
    cond_br %13, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %14 = "quantum.inst"(%3) {name = "z", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    br ^bb5
  ^bb4:  // pred: ^bb2
    %15 = "quantum.inst"(%3) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    %16 = alloc() : memref<2xi1>
    %17 = index_cast %c0_i64 : i64 to index
    %18 = "quantum.qextract"(%4, %17) : (!quantum.Array, index) -> !quantum.Qubit
    %19 = "quantum.inst"(%18) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %19, %16[%17] : memref<2xi1>
    %20 = index_cast %c1_i64 : i64 to index
    %21 = "quantum.qextract"(%4, %20) : (!quantum.Array, index) -> !quantum.Qubit
    %22 = "quantum.inst"(%21) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    store %22, %16[%20] : memref<2xi1>
    %23 = load %16[%c1_i64] : memref<2xi1>
    %24 = index_cast %c1_i64 : i64 to i1
    %25 = cmpi "eq", %23, %24 : i1
    cond_br %25, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %cst = constant 2.200000e+00 : f64
    %26 = "quantum.inst"(%3, %cst) {name = "ry", operand_segment_sizes = dense<1> : vector<2xi32>} : (!quantum.Qubit, f64) -> none
    br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %27 = load %16[%c0_i64] : memref<2xi1>
    %28 = index_cast %c1_i64 : i64 to i1
    %29 = cmpi "eq", %28, %27 : i1
    cond_br %29, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %cst_0 = constant 2.200000e+00 : f64
    %30 = "quantum.inst"(%3, %cst_0) {name = "ry", operand_segment_sizes = dense<1> : vector<2xi32>} : (!quantum.Qubit, f64) -> none
    br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    "quantum.dealloc"(%4) : (!quantum.Array) -> ()
    return
  }
  func @if_stmt(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_if_stmt() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(test, mlir);
}

TEST(qasm3VisitorTester, checkLoopStmt) {
  const std::string for_stmt = R"#(OPENQASM 3;
include "qelib1.inc";

for i in {11,22,33} {
    print(i);
}

for i in [0:10] {
    print(i);
    
}
for j in [0:2:4] {
    print("steps:", j);
}


for j in [0:4] {
    print("j in 0:4", j);
}

for i in [0:4] {
 for j in {1,2,3} {
     print(i,j);
 }
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", for_stmt, "for_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << "for_stmt MLIR:\n" << mlir << "\n";

  const std::string expected = R"#(module  {
  func @__internal_mlir_for_stmt() {
    %0 = alloc() : memref<3xi64>
    %c11_i64 = constant 11 : i64
    %c0_i64 = constant 0 : i64
    store %c11_i64, %0[%c0_i64] : memref<3xi64>
    %c22_i64 = constant 22 : i64
    %c1_i64 = constant 1 : i64
    store %c22_i64, %0[%c1_i64] : memref<3xi64>
    %c33_i64 = constant 33 : i64
    %c2_i64 = constant 2 : i64
    store %c33_i64, %0[%c2_i64] : memref<3xi64>
    %c0_i64_0 = constant 0 : i64
    %1 = index_cast %c0_i64_0 : i64 to index
    %2 = index_cast %c0_i64_0 : i64 to index
    %3 = alloca() : memref<1xi64>
    store %1, %3[%1] : memref<1xi64>
    %c3_i64 = constant 3 : i64
    %4 = index_cast %c3_i64 : i64 to index
    %c1_i64_1 = constant 1 : i64
    %5 = index_cast %c1_i64_1 : i64 to index
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %6 = load %3[%2] : memref<1xi64>
    %7 = cmpi "slt", %6, %4 : i64
    cond_br %7, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %8 = load %0[%6] : memref<3xi64>
    "quantum.print"(%8) : (i64) -> ()
    br ^bb3
  ^bb3:  // pred: ^bb2
    %9 = load %3[%2] : memref<1xi64>
    %10 = "std.addi"(%9, %5) : (i64, index) -> i64
    store %10, %3[%2] : memref<1xi64>
    br ^bb1
  ^bb4:  // pred: ^bb1
    %c0_i64_2 = constant 0 : i64
    %11 = index_cast %c0_i64_2 : i64 to index
    %12 = index_cast %c0_i64_2 : i64 to index
    %13 = alloca() : memref<1xi64>
    store %11, %13[%11] : memref<1xi64>
    %c10_i64 = constant 10 : i64
    %14 = index_cast %c10_i64 : i64 to index
    %c1_i64_3 = constant 1 : i64
    %15 = index_cast %c1_i64_3 : i64 to index
    %16 = load %13[%12] : memref<1xi64>
    br ^bb5
  ^bb5:  // 2 preds: ^bb4, ^bb7
    %17 = load %13[%12] : memref<1xi64>
    %18 = cmpi "slt", %17, %14 : i64
    cond_br %18, ^bb6, ^bb8
  ^bb6:  // pred: ^bb5
    %19 = load %13[%12] : memref<1xi64>
    "quantum.print"(%19) : (i64) -> ()
    br ^bb7
  ^bb7:  // pred: ^bb6
    %20 = load %13[%12] : memref<1xi64>
    %21 = "std.addi"(%20, %15) : (i64, index) -> i64
    store %21, %13[%12] : memref<1xi64>
    br ^bb5
  ^bb8:  // pred: ^bb5
    %c0_i64_4 = constant 0 : i64
    %22 = index_cast %c0_i64_4 : i64 to index
    %23 = index_cast %c0_i64_4 : i64 to index
    %24 = alloca() : memref<1xi64>
    store %22, %24[%22] : memref<1xi64>
    %c4_i64 = constant 4 : i64
    %25 = index_cast %c4_i64 : i64 to index
    %c2_i64_5 = constant 2 : i64
    %26 = index_cast %c2_i64_5 : i64 to index
    %27 = load %24[%23] : memref<1xi64>
    br ^bb9
  ^bb9:  // 2 preds: ^bb8, ^bb11
    %28 = load %24[%23] : memref<1xi64>
    %29 = cmpi "slt", %28, %25 : i64
    cond_br %29, ^bb10, ^bb12
  ^bb10:  // pred: ^bb9
    %30 = load %24[%23] : memref<1xi64>
    %31 = "quantum.createString"() {text = "steps:", varname = "__internal_string_literal__6481524378875777607"} : () -> !quantum.StringType
    "quantum.print"(%31, %30) : (!quantum.StringType, i64) -> ()
    br ^bb11
  ^bb11:  // pred: ^bb10
    %32 = load %24[%23] : memref<1xi64>
    %33 = "std.addi"(%32, %26) : (i64, index) -> i64
    store %33, %24[%23] : memref<1xi64>
    br ^bb9
  ^bb12:  // pred: ^bb9
    %c0_i64_6 = constant 0 : i64
    %34 = index_cast %c0_i64_6 : i64 to index
    %35 = index_cast %c0_i64_6 : i64 to index
    %36 = alloca() : memref<1xi64>
    store %34, %36[%34] : memref<1xi64>
    %c4_i64_7 = constant 4 : i64
    %37 = index_cast %c4_i64_7 : i64 to index
    %c1_i64_8 = constant 1 : i64
    %38 = index_cast %c1_i64_8 : i64 to index
    %39 = load %36[%35] : memref<1xi64>
    br ^bb13
  ^bb13:  // 2 preds: ^bb12, ^bb15
    %40 = load %36[%35] : memref<1xi64>
    %41 = cmpi "slt", %40, %37 : i64
    cond_br %41, ^bb14, ^bb16
  ^bb14:  // pred: ^bb13
    %42 = load %36[%35] : memref<1xi64>
    %43 = "quantum.createString"() {text = "j in 0:4", varname = "__internal_string_literal__984236603343773838"} : () -> !quantum.StringType
    "quantum.print"(%43, %42) : (!quantum.StringType, i64) -> ()
    br ^bb15
  ^bb15:  // pred: ^bb14
    %44 = load %36[%35] : memref<1xi64>
    %45 = "std.addi"(%44, %38) : (i64, index) -> i64
    store %45, %36[%35] : memref<1xi64>
    br ^bb13
  ^bb16:  // pred: ^bb13
    %c0_i64_9 = constant 0 : i64
    %46 = index_cast %c0_i64_9 : i64 to index
    %47 = index_cast %c0_i64_9 : i64 to index
    %48 = alloca() : memref<1xi64>
    store %46, %48[%46] : memref<1xi64>
    %c4_i64_10 = constant 4 : i64
    %49 = index_cast %c4_i64_10 : i64 to index
    %c1_i64_11 = constant 1 : i64
    %50 = index_cast %c1_i64_11 : i64 to index
    %51 = load %48[%47] : memref<1xi64>
    br ^bb17
  ^bb17:  // 2 preds: ^bb16, ^bb19
    %52 = load %48[%47] : memref<1xi64>
    %53 = cmpi "slt", %52, %49 : i64
    cond_br %53, ^bb18, ^bb20
  ^bb18:  // pred: ^bb17
    %54 = load %48[%47] : memref<1xi64>
    %55 = alloc() : memref<3xi64>
    store %c1_i64_11, %55[%c0_i64_9] : memref<3xi64>
    %c2_i64_12 = constant 2 : i64
    store %c2_i64_12, %55[%c1_i64_11] : memref<3xi64>
    %c3_i64_13 = constant 3 : i64
    store %c3_i64_13, %55[%c2_i64_12] : memref<3xi64>
    %c0_i64_14 = constant 0 : i64
    %56 = index_cast %c0_i64_14 : i64 to index
    %57 = index_cast %c0_i64_14 : i64 to index
    %58 = alloca() : memref<1xi64>
    store %56, %58[%56] : memref<1xi64>
    %c3_i64_15 = constant 3 : i64
    %59 = index_cast %c3_i64_15 : i64 to index
    %c1_i64_16 = constant 1 : i64
    %60 = index_cast %c1_i64_16 : i64 to index
    br ^bb21
  ^bb19:  // pred: ^bb24
    %61 = load %48[%47] : memref<1xi64>
    %62 = "std.addi"(%61, %50) : (i64, index) -> i64
    store %62, %48[%47] : memref<1xi64>
    br ^bb17
  ^bb20:  // pred: ^bb17
    return
  ^bb21:  // 2 preds: ^bb18, ^bb23
    %63 = load %58[%57] : memref<1xi64>
    %64 = cmpi "slt", %63, %59 : i64
    cond_br %64, ^bb22, ^bb24
  ^bb22:  // pred: ^bb21
    %65 = load %55[%63] : memref<3xi64>
    "quantum.print"(%54, %65) : (i64, i64) -> ()
    br ^bb23
  ^bb23:  // pred: ^bb22
    %66 = load %58[%57] : memref<1xi64>
    %67 = "std.addi"(%66, %60) : (i64, index) -> i64
    store %67, %58[%57] : memref<1xi64>
    br ^bb21
  ^bb24:  // pred: ^bb21
    br ^bb19
  }
  func @for_stmt(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_for_stmt() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected, mlir);

  const std::string for_stmt2 = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
bit result;

int[32] i = 0;
// Keep applying hadamards and measuring a qubit
// until 10, |1>s are measured
while (i < 10) {
    h q;
    result = measure q;
    if (result) {
        i++;
    }
}
)#";
  auto mlir2 = qcor::mlir_compile("qasm3", for_stmt2, "for_stmt2",
                                  qcor::OutputType::MLIR, false);
  std::cout << "for_stmt2 MLIR:\n" << mlir2 << "\n";
}

// TEST(qasm3VisitorTester, checkPrint) {
//   const std::string print_stmt = R"#(OPENQASM 3;
// include "qelib1.inc";
// int[32] i = 22;
// float[32] j = 2.2;
// print(i);
// print(j);
// print(i,j);
// print("hello world");
// print("can print with other vals", i, j);
// )#";
//   auto mlir = qcor::mlir_compile("qasm3", print_stmt, "print_stmt",
//                                  qcor::OutputType::MLIR, false);
//   std::cout << mlir << "\n";

//   // auto mlir2 = qcor::mlir_compile("qasm3", print_stmt, "print_stmt",
//   //                                 qcor::OutputType::LLVMMLIR, true);
//   // std::cout << mlir2 << "\n";

//   // auto mlir3 = qcor::mlir_compile("qasm3", print_stmt, "print_stmt",
//   //                                 qcor::OutputType::LLVMIR, true);
//   // std::cout << mlir3 << "\n";
// }


TEST(qasm3VisitorTester, checkSubroutine2) {
  const std::string subroutine_test = R"#(OPENQASM 3;
include "qelib1.inc";
const n = 10;
def xmeasure qubit:q -> bit { h q; return measure q; }
def ymeasure qubit:q -> bit { s q; h q; return measure q; }

def pauli_measurement(bit[2*n]:spec) qubit[n]:q -> bit {
  bit b = 0;
  for i in [0: n - 1] {
    bit temp;
    // FIXME WE HAVE TO WRAP EXPR IN () TO MAKE IT WORK
    if((spec[i]==1) && (spec[n+i]==0)) { temp = xmeasure q[i]; }
    if((spec[i]==0) && (spec[n+i]==1)) { temp = measure q[i]; }
    if((spec[i]==1) && (spec[n+i]==1)) { temp = ymeasure q[i]; }
    b ^= temp;
  }
  return b;
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", subroutine_test, "subroutine_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "subroutine_test MLIR:\n" << mlir << "\n";

  auto expected = R"#(module  {
  func @__internal_mlir_subroutine_test() {
    %c10_i64 = constant 10 : i64
  }
  func @subroutine_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_subroutine_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
  func @xmeasure(%arg0: !quantum.Qubit) -> i1 {
    %0 = "quantum.inst"(%arg0) {name = "h", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %1 = "quantum.inst"(%arg0) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    return %1 : i1
  }
  func @ymeasure(%arg0: !quantum.Qubit) -> i1 {
    %0 = "quantum.inst"(%arg0) {name = "s", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %1 = "quantum.inst"(%arg0) {name = "h", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %2 = "quantum.inst"(%arg0) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    return %2 : i1
  }
  func @pauli_measurement(%arg0: memref<20xi1>, %arg1: !quantum.Array) -> i1 {
    %c0_i64 = constant 0 : i64
    %0 = index_cast %c0_i64 : i64 to index
    %1 = alloca() : memref<1xi1>
    store %c0_i64, %1[%0] : memref<1xi1>
    %c0_i64_0 = constant 0 : i64
    %2 = index_cast %c0_i64_0 : i64 to index
    %3 = index_cast %c0_i64_0 : i64 to index
    %4 = alloca() : memref<1xi64>
    store %2, %4[%2] : memref<1xi64>
    %c9_i64 = constant 9 : i64
    %5 = index_cast %c9_i64 : i64 to index
    %c1_i64 = constant 1 : i64
    %6 = index_cast %c1_i64 : i64 to index
    %7 = load %4[%3] : memref<1xi64>
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb3
    %8 = load %4[%3] : memref<1xi64>
    %9 = cmpi "slt", %8, %5 : i64
    cond_br %9, ^bb2, ^bb4
  ^bb2:  // pred: ^bb1
    %10 = load %4[%3] : memref<1xi64>
    %11 = alloc() : memref<1xi1>
    %12 = load %arg0[%10] : memref<20xi1>
    %13 = index_cast %c1_i64 : i64 to i1
    %14 = cmpi "eq", %12, %13 : i1
    %15 = addi %c10_i64, %10 : i64
    %16 = load %arg0[%15] : memref<20xi1>
    %17 = index_cast %c0_i64_0 : i64 to i1
    %18 = cmpi "eq", %16, %17 : i1
    %19 = and %14, %18 : i1
    cond_br %19, ^bb5, ^bb6
  ^bb3:  // pred: ^bb10
    %20 = load %4[%3] : memref<1xi64>
    %21 = "std.addi"(%20, %6) : (i64, index) -> i64
    store %21, %4[%3] : memref<1xi64>
    br ^bb1
  ^bb4:  // pred: ^bb1
    %22 = index_cast %c0_i64 : i64 to index
    %23 = load %1[%22] : memref<1xi1>
    return %23 : i1
    return
  ^bb5:  // pred: ^bb2
    %24 = "quantum.qextract"(%arg1, %10) : (!quantum.Array, i64) -> !quantum.Qubit
    %25 = call @xmeasure(%24) : (!quantum.Qubit) -> i1
    %c0_i64_1 = constant 0 : i64
    store %25, %11[%c0_i64_1] : memref<1xi1>
    br ^bb6
  ^bb6:  // 2 preds: ^bb2, ^bb5
    %26 = load %arg0[%10] : memref<20xi1>
    %27 = index_cast %c0_i64_0 : i64 to i1
    %28 = cmpi "eq", %26, %27 : i1
    %29 = addi %c10_i64, %10 : i64
    %30 = load %arg0[%29] : memref<20xi1>
    %31 = index_cast %c1_i64 : i64 to i1
    %32 = cmpi "eq", %30, %31 : i1
    %33 = and %28, %32 : i1
    cond_br %33, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %34 = "quantum.qextract"(%arg1, %10) : (!quantum.Array, i64) -> !quantum.Qubit
    %35 = "quantum.inst"(%34) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %c0_i64_2 = constant 0 : i64
    %36 = index_cast %c0_i64_2 : i64 to index
    store %35, %11[%36] : memref<1xi1>
    br ^bb8
  ^bb8:  // 2 preds: ^bb6, ^bb7
    %37 = load %arg0[%10] : memref<20xi1>
    %38 = index_cast %c1_i64 : i64 to i1
    %39 = cmpi "eq", %37, %38 : i1
    %40 = addi %c10_i64, %10 : i64
    %41 = load %arg0[%40] : memref<20xi1>
    %42 = index_cast %c1_i64 : i64 to i1
    %43 = cmpi "eq", %41, %42 : i1
    %44 = and %39, %43 : i1
    cond_br %44, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %45 = "quantum.qextract"(%arg1, %10) : (!quantum.Array, i64) -> !quantum.Qubit
    %46 = call @ymeasure(%45) : (!quantum.Qubit) -> i1
    %c0_i64_3 = constant 0 : i64
    store %46, %11[%c0_i64_3] : memref<1xi1>
    br ^bb10
  ^bb10:  // 2 preds: ^bb8, ^bb9
    %false = constant false
    %47 = index_cast %false : i1 to index
    %48 = load %11[%47] : memref<1xi1>
    %49 = index_cast %c0_i64_0 : i64 to index
    %50 = load %1[%49] : memref<1xi1>
    %51 = xor %50, %48 : i1
    %52 = index_cast %c0_i64_0 : i64 to index
    store %51, %1[%52] : memref<1xi1>
    br ^bb3
  }
}
)#";
 EXPECT_EQ(expected, mlir);
}

TEST(qasm3VisitorTester, checkComplexCondition) {
  const std::string complex_condition = R"#(OPENQASM 3;
include "qelib1.inc";

const n = 20;
bit spec[10];

if((spec[0]==1) && (spec[n+2]==0)) { 
  print("hello");
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", complex_condition, "complex_condition",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
}

TEST(qasm3VisitorTester, checkUintIndexing) {
  const std::string complex_condition = R"#(OPENQASM 3;
include "qelib1.inc";

uint[4] b_in = 15;

bool b = bool(b_in[1]);

)#";
  auto mlir = qcor::mlir_compile("qasm3", complex_condition, "complex_condition",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
