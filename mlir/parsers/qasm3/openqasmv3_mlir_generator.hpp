#pragma once

#include "Quantum/QuantumOps.h"
#include "mlir_generator.hpp"

namespace qcor {
class qasm3_visitor;

class OpenQasmV3MLIRGenerator : public qcor::QuantumMLIRGenerator {
 protected:
  std::string file_name = "main";
  bool add_entry_point = true;
  bool add_custom_return = false;

  mlir::Type return_type;

  mlir::Type qubit_type;
  mlir::Type array_type;
  mlir::Type result_type;

  std::map<std::string, mlir::Value> global_symbol_table;
  bool add_main = true;
  std::shared_ptr<qasm3_visitor> visitor;

 public:
  OpenQasmV3MLIRGenerator(mlir::MLIRContext &context)
      : QuantumMLIRGenerator(context) {
    m_module = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  OpenQasmV3MLIRGenerator(mlir::OpBuilder b, mlir::MLIRContext &ctx)
      : QuantumMLIRGenerator(b, ctx) {
    m_module = mlir::ModuleOp::create(builder.getUnknownLoc());
  }

  void initialize_mlirgen(const std::string func_name,
                          std::vector<mlir::Type> arg_types,
                          std::vector<std::string> arg_var_names,
                          std::vector<std::string> var_attributes,
                          mlir::Type return_type);
  void initialize_mlirgen(
      bool add_entry_point = true, const std::string file_name = "",
      std::map<std::string, std::string> extra_quantum_args = {}) override;
  void mlirgen(const std::string &src) override;
  void finalize_mlirgen() override;
};
}  // namespace qcor