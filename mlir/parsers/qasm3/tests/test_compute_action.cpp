#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"
#include <fstream>

namespace {
// returns count of non-overlapping occurrences of 'sub' in 'str'
int countSubstring(const std::string &str, const std::string &sub) {
  if (sub.length() == 0) return 0;
  int count = 0;
  for (size_t offset = str.find(sub); offset != std::string::npos;
       offset = str.find(sub, offset + sub.length())) {
    ++count;
  }
  return count;
}
}  // namespace

TEST(qasm3ComputeActionTester, checkSimple) {
  const std::string src = R"#(OPENQASM 3;

qubit[4] q;
let bottom_three = q[1:3];

compute {
    rx(1.57) q[0];
    h bottom_three;
    for i in [0:3] {
      cnot q[i], q[i + 1];
    }
} action {
    rz(2.2) q[4];
}

)#";
  auto llvm = qcor::mlir_compile(src, "test", qcor::OutputType::LLVMIR, true);
  std::cout << "LLVM:\n" << llvm << "\n";
  // 2 rxs, 6 hs, 6 cnots, 1 rz + decls == 19
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 19);
}

TEST(qasm3ComputeActionTester, checkCtrlOpt) {
  const std::string src = R"#(OPENQASM 3;

qubit qq;
qubit rr;
qubit ss;
qubit vv;
qubit ww;

gate test22 q, r, s, v {
    compute {
        rx(1.57) q;
        h r;
        h s;
        h v;
        ctrl @ x q, r; // compiler translates to cnot
        cnot r, s;
        cnot s, v;
    } action {
        rz(2.2) v;
    }
}

ctrl @ test22 ww, qq, rr, ss, vv;

)#";

  auto llvm = qcor::mlir_compile(src, "test", qcor::OutputType::LLVMIR, true);
  std::cout << "LLVM:\n" << llvm << "\n";
  // 2 rxs, 6 hs, 6 cnots, 1 rz + decls == 19
  EXPECT_EQ(countSubstring(llvm, "__quantum__qis"), 19);

  // Execute and assert we have the single CRZ
  std::string tmp_filename = "__tmp__ctrl_test.xacc";
  qcor::execute(src, "test", 3, {{"qrt", "nisq"}, {"print_final_submission", tmp_filename}});
  std::ifstream ifs(tmp_filename);
  std::string content( (std::istreambuf_iterator<char>(ifs) ),
                       (std::istreambuf_iterator<char>()    ) );
  
  EXPECT_EQ(countSubstring(content, "CNOT"), 6);
  EXPECT_EQ(countSubstring(content, "H"), 6);
  EXPECT_EQ(countSubstring(content, "Rx"), 2);
  EXPECT_EQ(countSubstring(content, "CRZ"), 1);
  
  std::remove(tmp_filename.c_str());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}