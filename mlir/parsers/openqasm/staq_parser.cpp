
#include "staq_parser.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"

namespace qasm_parser {

StaqToMLIR::StaqToMLIR(mlir::MLIRContext &context) : builder(&context) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  // This is a generic function, the return type will be inferred later.
  // Arguments type are uniformly unranked tensors.
  llvm::SmallVector<mlir::Type, 4> arg_types;
  auto func_type = builder.getFunctionType(arg_types, llvm::None);
  auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), "main", func_type);
  mlir::FuncOp function(proto);
  auto &entryBlock = *function.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);
  theModule.push_back(function);
}

void StaqToMLIR::addReturn() {
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
}

void StaqToMLIR::visit(VarAccess &) {}
// Expressions
void StaqToMLIR::visit(BExpr &) {}
void StaqToMLIR::visit(UExpr &) {}
void StaqToMLIR::visit(PiExpr &) {}
void StaqToMLIR::visit(IntExpr &) {}
void StaqToMLIR::visit(RealExpr &r) {}
void StaqToMLIR::visit(VarExpr &v) {}
void StaqToMLIR::visit(ResetStmt &) {}
void StaqToMLIR::visit(IfStmt &) {}
void StaqToMLIR::visit(BarrierGate &) {}
void StaqToMLIR::visit(GateDecl &) {}
void StaqToMLIR::visit(OracleDecl &) {}
void StaqToMLIR::visit(RegisterDecl &d) {
  if (d.is_quantum()) {
    std::uint64_t size = d.size();
    auto name = d.id();

    auto pos = d.pos();
    auto line = pos.get_linenum();
    auto col = pos.get_column();
    auto fname =
        pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
    auto location =
        builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

    auto integer_type = builder.getI64Type();
    auto integer_attr = mlir::IntegerAttr::get(integer_type, size);

    auto str_attr = builder.getStringAttr(name);

    auto returntype = mlir::VectorType::get({size}, integer_type);

    auto allocation = builder.create<mlir::quantum::QallocOp>(
        location, returntype, integer_attr, str_attr);
    qubit_allocations.insert({name, allocation});
  }
}

void StaqToMLIR::visit(AncillaDecl &) {}
void StaqToMLIR::visit(Program &prog) {
  // Program body
  prog.foreach_stmt([this](auto &stmt) { stmt.accept(*this); });
}
void StaqToMLIR::visit(MeasureStmt &m) {
  auto pos = m.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("mz");
  // params
  auto dataType = mlir::VectorType::get({1}, builder.getF64Type());
  std::vector<double> v{0.0};
  auto params_arr_ref = llvm::makeArrayRef(v);
  mlir::DenseElementsAttr params_dataAttribute;

  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_var_name = m.q_arg().var();
  if (!qubit_allocations.count(qreg_var_name)) {
    // throw an error
  }

  auto qubits = qubit_allocations[qreg_var_name].qubits();

  std::uint64_t qidx = m.q_arg().offset().value();
  auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
  mlir::Value pos2 = builder.create<mlir::ConstantOp>(location, integer_attr);
  mlir::Value qbit_value =
      builder.create<mlir::vector::ExtractElementOp>(location, qubits, pos2);
  qubits_for_inst.push_back(qbit_value);

  builder.create<mlir::quantum::InstOp>(location, str_attr,
                                        llvm::makeArrayRef(qubits_for_inst),
                                        params_dataAttribute);
}
void StaqToMLIR::visit(UGate &u) {
  auto pos = u.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("u3");
  // params
  auto dataType = mlir::VectorType::get({3}, builder.getF64Type());
  std::vector<double> v{u.theta().constant_eval().value(),
                        u.phi().constant_eval().value(),
                        u.lambda().constant_eval().value()};
  auto params_arr_ref = llvm::makeArrayRef(v);
  auto params_dataAttribute =
      mlir::DenseElementsAttr::get(dataType, params_arr_ref);

  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_var_name = u.arg().var();
  if (!qubit_allocations.count(qreg_var_name)) {
    // throw an error
  }

  auto qubits = qubit_allocations[qreg_var_name].qubits();

  std::uint64_t qidx = u.arg().offset().value();
  auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
  mlir::Value pos2 = builder.create<mlir::ConstantOp>(location, integer_attr);
  mlir::Value ctrl_qbit_value =
      builder.create<mlir::vector::ExtractElementOp>(location, qubits, pos2);
  qubits_for_inst.push_back(ctrl_qbit_value);

  builder.create<mlir::quantum::InstOp>(location, str_attr,
                                        llvm::makeArrayRef(qubits_for_inst),
                                        params_dataAttribute);
}
void StaqToMLIR::visit(CNOTGate &g) {
  auto pos = g.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr("cx");

  // params
  // auto dataType = mlir::VectorType::get({1}, builder.getF64Type());
  // std::vector<double> v{0.0};
  // auto params_arr_ref = llvm::makeArrayRef(v);
  mlir::DenseElementsAttr params_dataAttribute;
  
  // ctrl qbits
  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_ctrl_var_name = g.ctrl().var();
  if (!qubit_allocations.count(qreg_ctrl_var_name)) {
    // throw an error
  }

  auto qubits = qubit_allocations[qreg_ctrl_var_name].qubits();

  std::uint64_t qidx = g.ctrl().offset().value();
  auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
  mlir::Value pos2 = builder.create<mlir::ConstantOp>(location, integer_attr);
  mlir::Value ctrl_qbit_value =
      builder.create<mlir::vector::ExtractElementOp>(location, qubits, pos2);
  qubits_for_inst.push_back(ctrl_qbit_value);

  // tgt qubit
  auto qreg_tgt_var_name = g.tgt().var();
  if (!qubit_allocations.count(qreg_tgt_var_name)) {
    // throw an error
  }

  auto tgt_qubits = qubit_allocations[qreg_tgt_var_name].qubits();

  std::uint64_t qidxt = g.tgt().offset().value();
  auto integer_attrt = mlir::IntegerAttr::get(builder.getI64Type(), qidxt);
  mlir::Value post = builder.create<mlir::ConstantOp>(location, integer_attrt);
  mlir::Value tgt_qbit_value = builder.create<mlir::vector::ExtractElementOp>(
      location, tgt_qubits, post);
  qubits_for_inst.push_back(tgt_qbit_value);

  builder.create<mlir::quantum::InstOp>(location, str_attr,
                                        llvm::makeArrayRef(qubits_for_inst),
                                        params_dataAttribute);
}
//   void visit(BarrierGate&) = 0;
void StaqToMLIR::visit(DeclaredGate &g) {
  auto pos = g.pos();
  auto line = pos.get_linenum();
  auto col = pos.get_column();
  auto fname =
      pos.get_filename().empty() ? "<builtin_str>" : pos.get_filename();
  auto location =
      builder.getFileLineColLoc(builder.getIdentifier(fname), line, col);

  auto str_attr = builder.getStringAttr(g.name());

  // params
  mlir::DenseElementsAttr params_dataAttribute;
  if (g.num_cargs()) {
    auto dataType =
        mlir::VectorType::get({g.num_cargs()}, builder.getF64Type());
    std::vector<double> v;  //{0.0};
    for (int i = 0; i < g.num_cargs(); i++) {
      v.push_back(g.carg(i).constant_eval().value());
    }
    auto params_arr_ref = llvm::makeArrayRef(v);
    params_dataAttribute =
        mlir::DenseElementsAttr::get(dataType, params_arr_ref);
  } 

  // qbits
  std::vector<mlir::Value> qubits_for_inst;
  for (int i = 0; i < g.num_qargs(); i++) {
    auto qreg_var_name = g.qarg(i).var();
    if (!qubit_allocations.count(qreg_var_name)) {
      // throw an error
    }

    auto qubits = qubit_allocations[qreg_var_name].qubits();

    std::uint64_t qidx = g.qarg(i).offset().value();
    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    mlir::Value qbit_value =
        builder.create<mlir::vector::ExtractElementOp>(location, qubits, pos);
    qubits_for_inst.push_back(qbit_value);
  }

  builder.create<mlir::quantum::InstOp>(location, str_attr,
                                        llvm::makeArrayRef(qubits_for_inst),
                                        params_dataAttribute);
}

}  // namespace qasm_parser