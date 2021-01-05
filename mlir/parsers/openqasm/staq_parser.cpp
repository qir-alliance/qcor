#include "staq_parser.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"

namespace qasm_parser {

void StaqToMLIR::visit(Program &prog) {
  prog.foreach_stmt([this](auto &stmt) { stmt.accept(*this); });
}

StaqToMLIR::StaqToMLIR(mlir::MLIRContext &context) : builder(&context) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  // This is a generic function, the return type will be inferred later.
  // Arguments type are uniformly unranked tensors.
  llvm::SmallVector<mlir::Type, 4> arg_types;
  auto func_type = builder.getFunctionType(arg_types, llvm::None);
  auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), "main", func_type);
  mlir::FuncOp function(proto);
  main_entry_point = function.addEntryBlock();
  auto &entryBlock = *main_entry_point;
  builder.setInsertionPointToStart(&entryBlock);
  theModule.push_back(function);
  function_names.push_back("main");

  llvm::StringRef qubit_type_name("Qubit"), array_type_name("Array"),
      result_type_name("Result");
  mlir::Identifier dialect = mlir::Identifier::get("quantum", &context);
  qubit_type = mlir::OpaqueType::get(dialect, qubit_type_name, &context);
  array_type = mlir::OpaqueType::get(dialect, array_type_name, &context);
  result_type = mlir::OpaqueType::get(dialect, result_type_name, &context);
}

void StaqToMLIR::addReturn() {
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc());

  std::vector<llvm::StringRef> tmp(function_names.begin(),
                                   function_names.end());

  auto function_names_datatype =
      mlir::VectorType::get({function_names.size()}, builder.getI64Type());
  auto function_names_ref = llvm::makeArrayRef(tmp);
  auto attrs = mlir::DenseStringElementsAttr::get(function_names_datatype,
                                                  function_names_ref);

  mlir::Identifier id =
      mlir::Identifier::get("quantum.internal_functions", builder.getContext());

  theModule.setAttrs(
      llvm::makeArrayRef({mlir::NamedAttribute(std::make_pair(id, attrs))}));
}

void StaqToMLIR::visit(GateDecl &gate_function) {
  auto name = gate_function.id();
  static std::vector<std::string> builtins{
      "u3", "u2",   "u1",  "cx",  "id",  "u0",  "x",   "y",  "z",
      "h",  "s",    "sdg", "t",   "tdg", "rx",  "ry",  "rz", "cz",
      "cy", "swap", "ch",  "ccx", "crz", "cu1", "cu2", "cu3"};
  if (std::find(builtins.begin(), builtins.end(), name) == builtins.end()) {
    std::vector<mlir::Type> arg_types;

    function_names.push_back(name);

    auto n_args = gate_function.q_params().size();
    for (std::size_t i = 0; i < n_args; i++) {
      arg_types.push_back(qubit_type);
    }

    auto func_type = builder.getFunctionType(arg_types, llvm::None);
    auto proto = mlir::FuncOp::create(builder.getUnknownLoc(), name, func_type);
    mlir::FuncOp function(proto);
    auto &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    auto arguments = entryBlock.getArguments();

    for (std::size_t i = 0; i < n_args; i++) {
      auto argument = arguments[i];
      auto arg_name = gate_function.q_params()[i];
      temporary_sub_kernel_args.insert({arg_name, argument});
    }
    in_sub_kernel = true;

    gate_function.foreach_stmt([this](Gate &g) { g.accept(*this); });

    in_sub_kernel = false;
    temporary_sub_kernel_args.clear();
    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    theModule.push_back(function);

    builder.setInsertionPointToStart(main_entry_point);
  }
}

void StaqToMLIR::visit(RegisterDecl &d) {
  if (d.is_quantum()) {
    std::int64_t size = d.size();
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
    auto allocation = builder.create<mlir::quantum::QallocOp>(
        location, array_type, integer_attr, str_attr);
    qubit_allocations.insert({name, allocation});
  }
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
  mlir::DenseElementsAttr params_dataAttribute;

  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_var_name = m.q_arg().var();
  if (!qubit_allocations.count(qreg_var_name)) {
    // throw an error
  }

  std::uint64_t qidx = m.q_arg().offset().value();
  auto qbit_key = std::make_pair(qreg_var_name, qidx);
  mlir::Value qbit_value;
  if (extracted_qubits.count(qbit_key)) {
    qbit_value = extracted_qubits[qbit_key];
  } else {
    auto qubits = qubit_allocations[qreg_var_name].qubits();

    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, qubit_type, qubits, pos);

    extracted_qubits.insert({std::make_pair(qreg_var_name, qidx), qbit_value});
  }
  qubits_for_inst.push_back(qbit_value);

  builder.create<mlir::quantum::InstOp>(location, result_type, str_attr,
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

  std::uint64_t qidx = u.arg().offset().value();
  auto qbit_key = std::make_pair(qreg_var_name, qidx);
  mlir::Value qbit_value;
  if (extracted_qubits.count(qbit_key)) {
    qbit_value = extracted_qubits[qbit_key];
  } else {
    auto qubits = qubit_allocations[qreg_var_name].qubits();

    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, qubit_type, qubits, pos);

    extracted_qubits.insert({std::make_pair(qreg_var_name, qidx), qbit_value});
  }
  qubits_for_inst.push_back(qbit_value);

  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qubits_for_inst), params_dataAttribute);
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
  mlir::DenseElementsAttr params_dataAttribute;

  // ctrl qbits
  std::vector<mlir::Value> qubits_for_inst;
  auto qreg_ctrl_var_name = g.ctrl().var();
  auto qreg_tgt_var_name = g.tgt().var();

  if (!qubit_allocations.count(qreg_ctrl_var_name)) {
    // throw an error
  }

  if (!qubit_allocations.count(qreg_tgt_var_name)) {
    // throw an error
  }

  // Get CTRL Qubit MLIR Value
  std::uint64_t ctrl_idx = g.ctrl().offset().value();
  auto ctrlqkey = std::make_pair(qreg_ctrl_var_name, ctrl_idx);

  mlir::Value ctl_qbit_value;
  if (extracted_qubits.count(ctrlqkey)) {
    ctl_qbit_value = extracted_qubits[ctrlqkey];
  } else {
    auto qubits = qubit_allocations[qreg_ctrl_var_name].qubits();

    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), ctrl_idx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    ctl_qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, qubit_type, qubits, pos);

    extracted_qubits.insert(
        {std::make_pair(qreg_ctrl_var_name, ctrl_idx), ctl_qbit_value});
  }
  qubits_for_inst.push_back(ctl_qbit_value);

  // Get Target Qubit MLIR Value
  std::uint64_t tgt_idx = g.tgt().offset().value();
  auto tgtqkey = std::make_pair(qreg_tgt_var_name, tgt_idx);

  mlir::Value tgt_qbit_value;
  if (extracted_qubits.count(tgtqkey)) {
    tgt_qbit_value = extracted_qubits[tgtqkey];
  } else {
    auto qubits = qubit_allocations[qreg_tgt_var_name].qubits();

    auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), tgt_idx);
    mlir::Value pos = builder.create<mlir::ConstantOp>(location, integer_attr);
    tgt_qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
        location, qubit_type, qubits, pos);

    extracted_qubits.insert(
        {std::make_pair(qreg_tgt_var_name, tgt_idx), tgt_qbit_value});
  }
  qubits_for_inst.push_back(tgt_qbit_value);

  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qubits_for_inst), params_dataAttribute);

  // std::uint64_t qidx = g.ctrl().offset().value();
  // auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
  // mlir::Value pos2 = builder.create<mlir::ConstantOp>(location,
  // integer_attr); mlir::Value ctrl_qbit_value =
  //     builder.create<mlir::vector::ExtractElementOp>(location, qubits, pos2);
  // qubits_for_inst.push_back(ctrl_qbit_value);

  // // tgt qubit
  // auto qreg_tgt_var_name = g.tgt().var();
  // if (!qubit_allocations.count(qreg_tgt_var_name)) {
  //   // throw an error
  // }

  // auto tgt_qubits = qubit_allocations[qreg_tgt_var_name].qubits();

  // std::uint64_t qidxt = g.tgt().offset().value();
  // auto integer_attrt = mlir::IntegerAttr::get(builder.getI64Type(), qidxt);
  // mlir::Value post = builder.create<mlir::ConstantOp>(location,
  // integer_attrt); mlir::Value tgt_qbit_value =
  // builder.create<mlir::vector::ExtractElementOp>(
  //     location, tgt_qubits, post);
  // qubits_for_inst.push_back(tgt_qbit_value);

  //   builder.create<mlir::quantum::InstOp>(location, str_attr,
  //                                         llvm::makeArrayRef(qubits_for_inst),
  //                                         params_dataAttribute);
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

    if (!in_sub_kernel) {
      std::uint64_t qidx = g.qarg(i).offset().value();
      auto qbit_key = std::make_pair(qreg_var_name, qidx);
      mlir::Value qbit_value;
      if (extracted_qubits.count(qbit_key)) {
        qbit_value = extracted_qubits[qbit_key];
      } else {
        auto qubits = qubit_allocations[qreg_var_name].qubits();

        auto integer_attr = mlir::IntegerAttr::get(builder.getI64Type(), qidx);
        mlir::Value pos =
            builder.create<mlir::ConstantOp>(location, integer_attr);
        qbit_value = builder.create<mlir::quantum::ExtractQubitOp>(
            location, qubit_type, qubits, pos);

        extracted_qubits.insert(
            {std::make_pair(qreg_var_name, qidx), qbit_value});
      }
      qubits_for_inst.push_back(qbit_value);
    } else {
      auto qubit_kernel_arg = temporary_sub_kernel_args[qreg_var_name];
      qubits_for_inst.push_back(qubit_kernel_arg);
    }
  }

  builder.create<mlir::quantum::InstOp>(
      location, mlir::NoneType::get(builder.getContext()), str_attr,
      llvm::makeArrayRef(qubits_for_inst), params_dataAttribute);
}

}  // namespace qasm_parser