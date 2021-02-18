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
// need to implement / test
//my_int = int[16](my_uint);
)#";
  auto var_mlir = qcor::mlir_compile("qasm3", var_test, "var_test",
                                     qcor::OutputType::MLIR, false);

  std::cout << "Var Test MLIR:\n" << var_mlir << "\n";
  const std::string expected_var = R"#(module  {
  func @__internal_mlir_var_test() {
    %c0_i64 = constant 0 : i64
    %c10_i64 = constant 10 : i64
    %0 = alloc() : memref<1xi32>
    store %c10_i64, %0[%c0_i64] : memref<1xi32>
    %cst = constant 0.000000e+00 : f32
    %1 = alloc() : memref<1xf32>
    store %cst, %1[%c0_i64] : memref<1xf32>
    %cst_0 = constant 3.140000e+00 : f64
    %2 = alloc() : memref<1xf64>
    store %cst_0, %2[%c0_i64] : memref<1xf64>
    %3 = "quantum.ralloc"() {name = "result", size = 1 : i64} : () -> !quantum.Array
    %4 = "quantum.cextract"(%3, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %5 = "quantum.ralloc"() {name = "results", size = 2 : i64} : () -> !quantum.Array
    %6 = "quantum.ralloc"() {name = "c", size = 22 : i64} : () -> !quantum.Array
    %false = constant false
    %7 = alloc() : memref<1xi1>
    store %false, %7[%c0_i64] : memref<1xi1>
    %true = constant true
    %8 = alloc() : memref<1xi1>
    store %true, %8[%c0_i64] : memref<1xi1>
    %false_1 = constant false
    %9 = alloc() : memref<1xi1>
    store %false_1, %9[%c0_i64] : memref<1xi1>
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
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
