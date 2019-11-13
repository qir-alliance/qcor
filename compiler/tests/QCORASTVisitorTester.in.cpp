#include <gtest/gtest.h>

#include "CommonGates.hpp"
#include "CountGatesOfTypeVisitor.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"

#include "clang/Parse/ParseAST.h"

#include "qcor_frontend_action.hpp"

#include <fstream>
#include <memory>
using namespace llvm;
using namespace clang;
using namespace qcor;
using namespace qcor::compiler;

const std::string rtimeCapture = R"rtimeCapture(#include <vector>
using qbit = std::vector<int>;
int main() {
    double angle = 3.1415;
    auto l = [&](qbit q, double t) {
        X(q[0]);
        Ry(q[1]);
        CX(q[1],q[0]);
        Rx(q[1],angle);
        Rx(q[1],-angle);
    };
    return 0;
})rtimeCapture";

TEST(QCORASTVisitorTester, checkSimple) {
  Rewriter rewriter1;
  xacc::setAccelerator("dummy");
  std::vector<std::string> args{"-std=c++14","-I@CMAKE_INSTALL_PREFIX@/include/xacc"};
  auto action1 = new QCORFrontendAction(rewriter1, "temp.cpp", args);
  const std::string bell = R"bell(#include <vector>
using qbit = std::vector<int>;
int main() {
    auto l = [&](qbit q) {
        H(q[0]);
        CX(q[0],q[1]);
        Measure(q[0]);
    };
    return 0;
})bell";

  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, bell, args));
  std::ifstream t(".temp_out.cpp");
  std::string src((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  std::remove(".temp_out.cpp");
  EXPECT_EQ(R"###(#include <vector>
using qbit = std::vector<int>;
int main() {
    auto l = [&](qbit q) {
std::istringstream iss(R"({
    "circuits": [
        {
            "circuit": "tmp_lambda",
            "instructions": [
                {
                    "gate": "H",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        0
                    ],
                    "parameters": []
                },
                {
                    "gate": "CNOT",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        0,
                        1
                    ],
                    "parameters": []
                },
                {
                    "gate": "Measure",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        0
                    ],
                    "parameters": [
                        0
                    ]
                }
            ],
            "variables": [],
            "coefficient": 1.0,
            "accelerator_signature": "dummy:"
        }
    ]
})");
auto function = xacc::getIRProvider("quantum")->createComposite("f");
function->load(iss);
function->expand({std::make_pair("param_id","t")});
if (qcor::__internal::executeKernel) {
auto acc = xacc::getAccelerator(function->accelerator_signature());
acc->execute(q,function);
}
std::stringstream ss;
function->persist(ss);
return ss.str();
}
;
    return 0;
})###",
            src);

  // Extract the CompositeInstruction JSON
  auto first = src.find("std::istringstream iss(R\"(");
  auto last = src.find("})\");");
  auto f = xacc::getIRProvider("quantum")->createComposite("tmp");
  std::istringstream iss(src.substr(first + 26, last - first - 25));
  f->load(iss);

  using namespace xacc::quantum;
  CountGatesOfTypeVisitor<Hadamard> h(f);
  CountGatesOfTypeVisitor<CNOT> cx(f);
  CountGatesOfTypeVisitor<Measure> m(f);

  EXPECT_EQ(1, h.countGates());
  EXPECT_EQ(1, cx.countGates());
  EXPECT_EQ(1, m.countGates());
}

TEST(QCORASTVisitorTester, checkParamAnsatzDouble) {
  Rewriter rewriter1;
  xacc::setAccelerator("dummy");
  std::vector<std::string> args{"-std=c++14", "-I@CMAKE_INSTALL_PREFIX@/include/xacc"};
  auto action1 = new QCORFrontendAction(rewriter1, "temp.cpp", args);

  const std::string bell = R"bell(#include <vector>
using qbit = std::vector<int>;
int main() {
    auto l = [&](qbit q, double t) {
        X(q[0]);
        Ry(q[1], t);
        CX(q[1],q[0]);
        Measure(q[0]);
    };
    return 0;
})bell";

  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, bell, args));
  std::ifstream t(".temp_out.cpp");
  std::string src((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  std::remove(".temp_out.cpp");
  EXPECT_EQ(R"###(#include <vector>
using qbit = std::vector<int>;
int main() {
    auto l = [&](qbit q, double t) {
std::istringstream iss(R"({
    "circuits": [
        {
            "circuit": "tmp_lambda",
            "instructions": [
                {
                    "gate": "X",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        0
                    ],
                    "parameters": []
                },
                {
                    "gate": "Ry",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        1
                    ],
                    "parameters": [
                        "t"
                    ]
                },
                {
                    "gate": "CNOT",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        1,
                        0
                    ],
                    "parameters": []
                },
                {
                    "gate": "Measure",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        0
                    ],
                    "parameters": [
                        0
                    ]
                }
            ],
            "variables": [
                "t"
            ],
            "coefficient": 1.0,
            "accelerator_signature": "dummy:"
        }
    ]
})");
auto function = xacc::getIRProvider("quantum")->createComposite("f");
function->load(iss);
function->expand({std::make_pair("param_id","t")});
if (qcor::__internal::executeKernel) {
auto acc = xacc::getAccelerator(function->accelerator_signature());
std::vector<double> params{t};
function = function->operator()(params);
acc->execute(q,function);
}
std::stringstream ss;
function->persist(ss);
return ss.str();
}
;
    return 0;
})###",
            src);

  // Extract the CompositeInstruction JSON
  auto first = src.find("std::istringstream iss(R\"(");
  auto last = src.find("})\");");
  auto f = xacc::getIRProvider("quantum")->createComposite("tmp");
  std::istringstream iss(src.substr(first + 26, last - first - 25));
  f->load(iss);

  using namespace xacc::quantum;
  CountGatesOfTypeVisitor<X> x(f);
  CountGatesOfTypeVisitor<Ry> ry(f);
  CountGatesOfTypeVisitor<CNOT> cx(f);
  CountGatesOfTypeVisitor<Measure> m(f);

  EXPECT_EQ(1, x.countGates());
  EXPECT_EQ(1, ry.countGates());
  EXPECT_EQ(1, cx.countGates());
  EXPECT_EQ(1, m.countGates());

  auto evaled = f->operator()({2.2});
  EXPECT_EQ(2.2, evaled->getInstruction(1)->getParameter(0).as<double>());
}

TEST(QCORASTVisitorTester, checkParamAnsatzVectorDouble) {
  Rewriter rewriter1;
  xacc::setAccelerator("dummy");
  std::vector<std::string> args{"-std=c++14","-I@CMAKE_INSTALL_PREFIX@/include/xacc"};
  auto action1 = new QCORFrontendAction(rewriter1, "temp.cpp", args);

  const std::string bell = R"bell(#include <vector>
using qbit = std::vector<int>;
int main() {
    auto l = [&](qbit q, std::vector<double> t) {
        X(q[0]);
        Ry(q[1], t[0]);
        CX(q[1],q[0]);
        Measure(q[0]);
    };
    return 0;
})bell";

  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, bell, args));
  std::ifstream t(".temp_out.cpp");
  std::string src((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  std::remove(".temp_out.cpp");
  EXPECT_EQ(R"###(#include <vector>
using qbit = std::vector<int>;
int main() {
    auto l = [&](qbit q, std::vector<double> t) {
std::istringstream iss(R"({
    "circuits": [
        {
            "circuit": "tmp_lambda",
            "instructions": [
                {
                    "gate": "X",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        0
                    ],
                    "parameters": []
                },
                {
                    "gate": "Ry",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        1
                    ],
                    "parameters": [
                        "t0"
                    ]
                },
                {
                    "gate": "CNOT",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        1,
                        0
                    ],
                    "parameters": []
                },
                {
                    "gate": "Measure",
                    "enabled": true,
                    "composite": false,
                    "qubits": [
                        0
                    ],
                    "parameters": [
                        0
                    ]
                }
            ],
            "variables": [
                "t0"
            ],
            "coefficient": 1.0,
            "accelerator_signature": "dummy:"
        }
    ]
})");
auto function = xacc::getIRProvider("quantum")->createComposite("f");
function->load(iss);
function->expand({std::make_pair("param_id","t")});
if (qcor::__internal::executeKernel) {
auto acc = xacc::getAccelerator(function->accelerator_signature());
std::vector<double> params{t};
function = function->operator()(params);
acc->execute(q,function);
}
std::stringstream ss;
function->persist(ss);
return ss.str();
}
;
    return 0;
})###",
            src);

  // Extract the CompositeInstruction JSON
  auto first = src.find("std::istringstream iss(R\"(");
  auto last = src.find("})\");");
  auto f = xacc::getIRProvider("quantum")->createComposite("tmp");
  std::istringstream iss(src.substr(first + 26, last - first - 25));
  f->load(iss);

  using namespace xacc::quantum;
  CountGatesOfTypeVisitor<X> x(f);
  CountGatesOfTypeVisitor<Ry> ry(f);
  CountGatesOfTypeVisitor<CNOT> cx(f);
  CountGatesOfTypeVisitor<Measure> m(f);

  EXPECT_EQ(1, x.countGates());
  EXPECT_EQ(1, ry.countGates());
  EXPECT_EQ(1, cx.countGates());
  EXPECT_EQ(1, m.countGates());

  auto evaled = f->operator()({2.2});
  EXPECT_EQ(2.2, evaled->getInstruction(1)->getParameter(0).as<double>());
}

// TEST(QCORASTVisitorTester, checkHWEGenerator) {
//   Rewriter rewriter1;
//   xacc::setAccelerator("dummy");
//   auto action1 = new QCORFrontendAction(rewriter1, "temp.cpp");
//   std::vector<std::string> args{"-std=c++14",
//                                 "-I@CMAKE_INSTALL_PREFIX@/include/qcor",
//                                 "-I@CMAKE_INSTALL_PREFIX@/include/xacc"};

//   const std::string bell = R"hwe(#include "qcor.hpp"
// int main() {
//     std::vector<std::pair<int,int>> c{{0,1},{1,2},{1,3}};
//     auto l = [&](qbit q, std::vector<double> x) {
//         hwe(q, x, {{"nq", 4}, {"layers", 2}, {"coupling", c}});
//         Measure(q[0]);
//     };
//     return 0;
// })hwe";

//   EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, bell, args));
//   std::ifstream t(".temp_out.cpp");
//   std::string src((std::istreambuf_iterator<char>(t)),
//                   std::istreambuf_iterator<char>());
//   std::remove(".temp_out.cpp");
//   EXPECT_EQ(R"###(#include "qcor.hpp"
// int main() {
//     std::vector<std::pair<int,int>> c{{0,1},{1,2},{1,3}};
//     auto l = [&](qbit q, std::vector<double> x) {
// std::istringstream iss(R"({
//     "circuits": [
//         {
//             "circuit": "tmp_lambda",
//             "instructions": [
//                 {
//                     "gate": "hwe",
//                     "enabled": true,
//                     "composite": true,
//                     "qubits": [],
//                     "parameters": [],
//                     "variables": []
//                 },
//                 {
//                     "gate": "Measure",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": [
//                         0
//                     ]
//                 }
//             ],
//             "variables": [],
//             "coefficient": 1.0,
//             "accelerator_signature": "dummy:"
//         }
//     ]
// })");
// auto function = xacc::getIRProvider("quantum")->createComposite("f");
// function->load(iss);
// function->expand({std::make_pair("layers",2),std::make_pair("nq",4),std::make_pair("coupling",c),std::make_pair("param_id","x")});
// if (qcor::__internal::executeKernel) {
// auto acc = xacc::getAccelerator(function->accelerator_signature());
// std::vector<double> params{x};
// function = function->operator()(params);
// acc->execute(q,function);
// }
// std::stringstream ss;
// function->persist(ss);
// return ss.str();
// }
// ;
//     return 0;
// })###",
//             src);

//   // Extract the CompositeInstruction JSON
//   auto first = src.find("std::istringstream iss(R\"(");
//   auto last = src.find("})\");");
//   auto f = xacc::getIRProvider("quantum")->createComposite("tmp");
//   std::istringstream iss(src.substr(first + 26, last - first - 25));
//   f->load(iss);

//   std::vector<std::pair<int, int>> c{{0, 1}, {1, 2}, {1, 3}};
//   EXPECT_TRUE(f->expand(xacc::HeterogeneousMap{std::make_pair("nq", 4),
//                                                std::make_pair("layers", 2),
//                                                std::make_pair("coupling", c)}));
//   EXPECT_EQ(32, f->nVariables());
//   EXPECT_EQ(39, 1+std::dynamic_pointer_cast<xacc::CompositeInstruction>(f->getInstruction(0))->nInstructions());

//   using namespace xacc::quantum;
//   CountGatesOfTypeVisitor<Rx> rx(f);
//   CountGatesOfTypeVisitor<Rz> rz(f);
//   CountGatesOfTypeVisitor<CNOT> cx(f);
//   CountGatesOfTypeVisitor<Measure> m(f);

//   EXPECT_EQ(12, rx.countGates());
//   EXPECT_EQ(20, rz.countGates());
//   EXPECT_EQ(6, cx.countGates());
//   EXPECT_EQ(1, m.countGates());
// }


// TEST(QCORASTVisitorTester, checkExpGenerator) {
//   Rewriter rewriter1;
//   xacc::setAccelerator("dummy");
//   auto action1 = new QCORFrontendAction(rewriter1, "temp.cpp");
//   std::vector<std::string> args{"-std=c++14",
//                                 "-I@CMAKE_INSTALL_PREFIX@/include/qcor",
//                                 "-I@CMAKE_INSTALL_PREFIX@/include/xacc"};

//   const std::string bell = R"hwe(#include "qcor.hpp"
// int main() {
//     auto l = [&](qbit q, double t0, double t1) {
//         X(q[0]);
//         exp_i_theta(q, t0, {{"pauli", "X0 Y1 - Y0 X1"}});
//         exp_i_theta(q, t1, {{"pauli", "X0 Z1 Y2 - X2 Z1 Y0"}});
//     };
//     return 0;
// })hwe";

//   EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, bell, args));
//   std::ifstream t(".temp_out.cpp");
//   std::string src((std::istreambuf_iterator<char>(t)),
//                   std::istreambuf_iterator<char>());
//   std::remove(".temp_out.cpp");
//   EXPECT_EQ(R"###(#include "qcor.hpp"
// int main() {
//     auto l = [&](qbit q, double t0, double t1) {
// std::istringstream iss(R"({
//     "circuits": [
//         {
//             "circuit": "tmp_lambda",
//             "instructions": [
//                 {
//                     "gate": "X",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1
//                     ],
//                     "parameters": [
//                         1.5707963267948966
//                     ]
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rz",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1
//                     ],
//                     "parameters": [
//                         "1.000000 * t0"
//                     ]
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1
//                     ],
//                     "parameters": [
//                         -1.5707963267948966
//                     ]
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": [
//                         1.5707963267948966
//                     ]
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rz",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1
//                     ],
//                     "parameters": [
//                         "-1.000000 * t0"
//                     ]
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": [
//                         -1.5707963267948966
//                     ]
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         2
//                     ],
//                     "parameters": [
//                         1.5707963267948966
//                     ]
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1,
//                         2
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rz",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         2
//                     ],
//                     "parameters": [
//                         "1.000000 * t1"
//                     ]
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1,
//                         2
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         2
//                     ],
//                     "parameters": [
//                         -1.5707963267948966
//                     ]
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         2
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": [
//                         1.5707963267948966
//                     ]
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1,
//                         2
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rz",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         2
//                     ],
//                     "parameters": [
//                         "-1.000000 * t1"
//                     ]
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         1,
//                         2
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "CNOT",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0,
//                         1
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "H",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         2
//                     ],
//                     "parameters": []
//                 },
//                 {
//                     "gate": "Rx",
//                     "enabled": true,
//                     "composite": false,
//                     "qubits": [
//                         0
//                     ],
//                     "parameters": [
//                         -1.5707963267948966
//                     ]
//                 }
//             ],
//             "variables": [
//                 "t1",
//                 "t0"
//             ],
//             "coefficient": 1.0,
//             "accelerator_signature": "dummy:"
//         }
//     ]
// })");
// auto function = xacc::getIRProvider("quantum")->createComposite("f");
// function->load(iss);
// function->expand({std::make_pair("param_id","t1"),std::make_pair("pauli","X0 Z1 Y2 - X2 Z1 Y0")});
// if (qcor::__internal::executeKernel) {
// auto acc = xacc::getAccelerator(function->accelerator_signature());
// std::vector<double> params{t0,t1};
// function = function->operator()(params);
// acc->execute(q,function);
// }
// std::stringstream ss;
// function->persist(ss);
// return ss.str();
// }
// ;
//     return 0;
// })###",
//             src);

//   // Extract the CompositeInstruction JSON
//   auto first = src.find("std::istringstream iss(R\"(");
//   auto last = src.find("})\");");
//   auto f = xacc::getIRProvider("quantum")->createComposite("tmp");
//   std::istringstream iss(src.substr(first + 26, last - first - 25));
//   f->load(iss);

//   EXPECT_EQ(2, f->nVariables());
//   EXPECT_EQ(33, f->nInstructions());

//   using namespace xacc::quantum;
//   CountGatesOfTypeVisitor<Rx> rx(f);
//   CountGatesOfTypeVisitor<Rz> rz(f);
//   CountGatesOfTypeVisitor<CNOT> cx(f);

//   EXPECT_EQ(8, rx.countGates());
//   EXPECT_EQ(4, rz.countGates());
//   EXPECT_EQ(12, cx.countGates());
// }

using namespace xacc;
class DummyAccelerator : public xacc::Accelerator {
public:
  const std::string name() const override { return "dummy"; }
  const std::string description() const override { return ""; }
  void initialize(const HeterogeneousMap &params = {}) override { return; }
  void execute(std::shared_ptr<xacc::AcceleratorBuffer> buf,
               std::shared_ptr<xacc::CompositeInstruction> f) override {}
  void execute(std::shared_ptr<AcceleratorBuffer> buffer,
               const std::vector<std::shared_ptr<CompositeInstruction>>
                   functions) override {}
  void updateConfiguration(const HeterogeneousMap &config) override {}
  const std::vector<std::string> configurationKeys() override { return {}; }
};

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  std::shared_ptr<Accelerator> acc = std::make_shared<DummyAccelerator>();
  xacc::contributeService("dummy", acc);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
