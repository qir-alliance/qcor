#include "SimplifyQubitExtractPass.hpp"
#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

namespace qcor {
void SimplifyQubitExtractPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}
void SimplifyQubitExtractPass::runOnOperation() {
  // Extract qubit op simplification will respect the regions:
  // e.g., for loops are flattened => can simplify
  // but not if blocks.
  using extract_qubit_map_type =
      std::unordered_map<mlir::Operation *,
                         std::unordered_map<int64_t, mlir::Value>>;
  std::unordered_map<mlir::Region *, extract_qubit_map_type>
      region_scoped_extract_qubit_map;

  // Map const qubit extract to its first extract
  getOperation().walk([&](mlir::quantum::ExtractQubitOp op) {
    mlir::Value idx_val = op.idx();
    mlir::Value qreg = op.qreg();
    mlir::Operation *qreg_create_op = qreg.getDefiningOp();
    if (qreg_create_op) {
      mlir::Region *region = op->getBlock()->getParent();
      if (region_scoped_extract_qubit_map.find(region) ==
          region_scoped_extract_qubit_map.end()) {
        region_scoped_extract_qubit_map[region] = {};
      }
      auto &extract_qubit_map = region_scoped_extract_qubit_map[region];
      if (extract_qubit_map.find(qreg_create_op) == extract_qubit_map.end()) {
        extract_qubit_map[qreg_create_op] = {};
      }

      auto idx_def_op = idx_val.getDefiningOp();
      if (idx_def_op) {
        // Try cast:
        if (auto const_def_op =
                dyn_cast_or_null<mlir::ConstantIntOp>(idx_def_op)) {
          // std::cout << "Constant extract index " << const_def_op.getValue()
          //           << "\n";
          const int64_t index_const = const_def_op.getValue();
          auto &previous_qreg_extract = extract_qubit_map[qreg_create_op];
          if (previous_qreg_extract.find(index_const) ==
              previous_qreg_extract.end()) {
            // std::cout << "First use\n";
            previous_qreg_extract[index_const] = op.qbit();
          } else {
            mlir::Value previous_extract = previous_qreg_extract[index_const];
            op.qbit().replaceAllUsesWith(previous_extract);
          }

          // Erase the extract cache in the parent scope as well:
          // i.e., when the child scope (e.g., if block) is accessing this
          // qubit (doing an extract), don't extend the use-def chain in the
          // parent scope passing this point.
          auto parent_op = region->getParentOp();
          if (parent_op) {
            if (auto parent_region = parent_op->getBlock()->getParent()) {
              if (region_scoped_extract_qubit_map.find(parent_region) !=
                  region_scoped_extract_qubit_map.end()) {
                auto &parent_region_map =
                    region_scoped_extract_qubit_map[parent_region];
                if (parent_region_map.find(qreg_create_op) !=
                    parent_region_map.end()) {
                  auto &parent_qreg_extract_map =
                      parent_region_map[qreg_create_op];
                  parent_qreg_extract_map.erase(index_const);
                }
              }
            }
          }
        }
      }
    }
  });

  // Fix up the SSA chain
  // Mini symbol table to track all SSA values.
  std::unordered_map<void *, void *> ssa_var_to_root;
  std::unordered_map<void *, mlir::Value> root_ssa_var_to_last_use;
  getOperation().walk([&](mlir::quantum::ValueSemanticsInstOp op) {
    if (op.qubits().size() == 1) {
      mlir::Value operand = op.qubits()[0];
      void *operand_ptr = operand.getAsOpaquePointer();
      if (ssa_var_to_root.find(operand_ptr) == ssa_var_to_root.end()) {
        ssa_var_to_root[operand_ptr] = operand_ptr;
        assert(root_ssa_var_to_last_use.find(operand_ptr) ==
               root_ssa_var_to_last_use.end());
        root_ssa_var_to_last_use[operand_ptr] = op.result()[0];
        ssa_var_to_root[op.result()[0].getAsOpaquePointer()] = operand_ptr;
      } else {
        // Match SSA operand:
        void *root_value_ptr = ssa_var_to_root[operand_ptr];
        assert(root_ssa_var_to_last_use.find(root_value_ptr) !=
               root_ssa_var_to_last_use.end());

        // Fix up the input operand and update the last output
        op.qubitsMutable().assign(root_ssa_var_to_last_use[root_value_ptr]);
        ssa_var_to_root[op.result()[0].getAsOpaquePointer()] = root_value_ptr;
        root_ssa_var_to_last_use[root_value_ptr] = op.result()[0];
      }
    } else {
      assert(op.qubits().size() == 2);
      std::vector<mlir::Value> new_operands{op.qubits()[0], op.qubits()[1]};
      for (int i = 0; i < 2; ++i) {
        mlir::Value operand = op.qubits()[i];
        void *operand_ptr = operand.getAsOpaquePointer();
        if (ssa_var_to_root.find(operand_ptr) == ssa_var_to_root.end()) {
          ssa_var_to_root[operand_ptr] = operand_ptr;
          assert(root_ssa_var_to_last_use.find(operand_ptr) ==
                 root_ssa_var_to_last_use.end());
          root_ssa_var_to_last_use[operand_ptr] = op.result()[i];
          ssa_var_to_root[op.result()[i].getAsOpaquePointer()] = operand_ptr;
        } else {
          // Match SSA operand:
          // Fix up the input operand and update the last output
          void *root_value_ptr = ssa_var_to_root[operand_ptr];
          assert(root_ssa_var_to_last_use.find(root_value_ptr) !=
                 root_ssa_var_to_last_use.end());
          new_operands[i] = root_ssa_var_to_last_use[root_value_ptr];
          ssa_var_to_root[op.result()[i].getAsOpaquePointer()] = root_value_ptr;
          root_ssa_var_to_last_use[root_value_ptr] = op.result()[i];
        }
      }
      op.qubitsMutable().assign(llvm::makeArrayRef(new_operands));
    }
  });
}
} // namespace qcor