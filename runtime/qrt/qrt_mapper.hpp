
#ifndef QCOR_QRT_MAPPER_HPP_
#define QCOR_QRT_MAPPER_HPP_

#include "AllGateVisitor.hpp"
#include "Circuit.hpp"
#include <Instruction.hpp>

#include "qrt.hpp"

namespace qcor {
using namespace xacc::quantum;

class qrt_mapper : public AllGateVisitor,
                   public xacc::InstructionVisitor<Circuit> {
protected:
  std::stringstream ss;
  // The kernel name of the CompositeInstruction
  // that this mapper is visiting.
  std::string kernelName;
  void addOneQubitGate(const std::string name, xacc::Instruction &inst) {
    auto expr = inst.getBitExpression(0);
    ss << "quantum::" + name + "(" << inst.getBufferNames()[0] << "["
       << (expr.empty() ? std::to_string(inst.bits()[0]) : expr) << "]";
    if (inst.isParameterized() && inst.name() != "Measure") {
      ss << ", " << inst.getParameter(0).toString();
      for (int i = 1; i < inst.nParameters(); i++) {
        ss << ", " << inst.getParameter(i).toString() << "\n";
      }
    }
    ss << ");\n";
  }

  void addTwoQubitGate(const std::string name, xacc::Instruction &inst) {
    auto expr_src = inst.getBitExpression(0);
    auto expr_tgt = inst.getBitExpression(1);
    ss << "quantum::" + name + "(" << inst.getBufferNames()[0] << "["
       << (expr_src.empty() ? std::to_string(inst.bits()[0]) : expr_src)
       << "], " << inst.getBufferNames()[1] << "["
       << (expr_tgt.empty() ? std::to_string(inst.bits()[1]) : expr_tgt) << "]";
    // Handle parameterized gate:
    if (inst.isParameterized()) {
      ss << ", " << inst.getParameter(0).toString();
      for (int i = 1; i < inst.nParameters(); i++) {
        ss << ", " << inst.getParameter(i).toString();
      }
    }
    ss << ");\n";
  }

public:
  // Ctor: cache the kernel name of the CompositeInstruction
  qrt_mapper(const std::string &top_level_kernel_name)
      : kernelName(top_level_kernel_name) {}

  auto get_new_src() { return ss.str(); }
  // One-qubit gates
  void visit(Hadamard &h) override { addOneQubitGate("h", h); }
  void visit(Rz &rz) override { addOneQubitGate("rz", rz); }
  void visit(Ry &ry) override { addOneQubitGate("ry", ry); }
  void visit(Rx &rx) override { addOneQubitGate("rx", rx); }
  void visit(X &x) override { addOneQubitGate("x", x); }
  void visit(Y &y) override { addOneQubitGate("y", y); }
  void visit(Z &z) override { addOneQubitGate("z", z); }
  void visit(S &s) override { addOneQubitGate("s", s); }
  void visit(Sdg &sdg) override { addOneQubitGate("sdg", sdg); }
  void visit(T &t) override { addOneQubitGate("t", t); }
  void visit(Tdg &tdg) override { addOneQubitGate("tdg", tdg); }

  // Two-qubit gates
  void visit(CNOT &cnot) override { addTwoQubitGate("cnot", cnot); }
  void visit(CY &cy) override { addTwoQubitGate("cy", cy); }
  void visit(CZ &cz) override { addTwoQubitGate("cz", cz); }
  void visit(Swap &s) override { addTwoQubitGate("swap", s); }
  void visit(CRZ &crz) override { addTwoQubitGate("crz", crz); }
  void visit(CH &ch) override { addTwoQubitGate("ch", ch); }
  void visit(CPhase &cphase) override { addTwoQubitGate("cphase", cphase); }

  void visit(Measure &measure) override { addOneQubitGate("mz", measure); }
  void visit(Identity &i) override { addOneQubitGate("i", i); }
  void visit(U &u) override { addOneQubitGate("u", u); }
  void visit(U1 &u1) override { addOneQubitGate("u1", u1); }
  void visit(Circuit &circ) override {
    if (circ.name() == kernelName) {
      return;
    }
    if (circ.name() == "exp_i_theta") {
      ss << "quantum::exp(" << circ.getBufferNames()[0] << ", "
         << circ.getArguments()[0]->name << ", " << circ.getArguments()[1]->name
         << ");\n";
    } else {
      // Call a previously-defined QCOR kernel:
      // In this context, we disable __execute flag around this hence
      // this sub-kernel will not be submitted.
      // i.e. only the outer-most kernel will be submitted.
      // Open a new scope since we use a local var '__cached_execute_flag'
      ss << "{\n";
      // Cache the state of the __execute flag
      ss << "const auto __cached_execute_flag = __execute;\n";
      // Reset the flag:
      ss << "__execute = false;\n";
      for (const auto &arg : circ.getArguments()) {
        if (arg->name == "__xacc__literal_") {
          // double nameMEMORYLOC = ...
          ss << arg->type << " " << arg->name << arg << " = ";
          // can be int or double
          if (arg->runtimeValue.keyExists<int>(
                  xacc::INTERNAL_ARGUMENT_VALUE_KEY)) {
            ss << arg->runtimeValue.get<int>(xacc::INTERNAL_ARGUMENT_VALUE_KEY)
               << ";\n";
          } else {
            ss << arg->runtimeValue.get<double>(
                      xacc::INTERNAL_ARGUMENT_VALUE_KEY)
               << ";\n";
          }
        }
      }
      // Add the circuit invocation.
      ss << circ.name() << "(" << circ.getBufferNames()[0];
      for (const auto &arg : circ.getArguments()) {
        if (arg->name.find("__xacc__literal_") != std::string::npos) {
          ss << ", " << arg->name << arg;
        } else {
          ss << ", " << arg->name;
        }
      }
      ss << ")"
         << ";\n";
      // Reinstate the __execute flag
      ss << "__execute = __cached_execute_flag;\n";
      ss << "}\n";
    }
  }
  void visit(IfStmt &ifStmt) override {}
};
} // namespace qcor
#endif