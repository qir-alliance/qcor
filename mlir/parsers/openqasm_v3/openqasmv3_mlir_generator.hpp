#pragma once

#include "Quantum/QuantumOps.h"
#include "mlir_generator.hpp"

namespace qcor {
class qasm3_visitor;

class OpenQasmV3MLIRGenerator : public qcor::QuantumMLIRGenerator {
 protected:
  std::string file_name = "main";
  bool add_entry_point = true;
  mlir::Type qubit_type;
  mlir::Type array_type;
  mlir::Type result_type;

  std::map<std::string, mlir::Value> global_symbol_table;
  bool add_main = true;
  std::shared_ptr<qasm3_visitor> visitor;

 public:
  OpenQasmV3MLIRGenerator(mlir::MLIRContext &context)
      : QuantumMLIRGenerator(context) {}
  void initialize_mlirgen(bool add_entry_point = true,
                          const std::string file_name = "") override;
  void mlirgen(const std::string &src) override;
  void finalize_mlirgen() override;
};
}  // namespace qcor