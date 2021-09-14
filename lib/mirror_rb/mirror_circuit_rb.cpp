#include "mirror_circuit_rb.hpp"
#include "clifford_gate_utils.hpp"
#include "qcor_ir.hpp"
#include "qcor_pimpl_impl.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <cassert>
#include <random>

namespace {
std::vector<std::shared_ptr<xacc::Instruction>>
getLayer(std::shared_ptr<xacc::CompositeInstruction> circuit, int layerId) {
  std::vector<std::shared_ptr<xacc::Instruction>> result;
  assert(layerId < circuit->depth());
  auto graphView = circuit->toGraph();
  for (int i = 1; i < graphView->order() - 1; i++) {
    auto node = graphView->getVertexProperties(i);
    if (node.get<int>("layer") == layerId) {
      result.emplace_back(
          circuit->getInstruction(node.get<std::size_t>("id") - 1)->clone());
    }
  }
  assert(!result.empty());
  return result;
}
} // namespace

namespace qcor {
std::pair<std::shared_ptr<CompositeInstruction>, std::vector<bool>>
createMirrorCircuit(std::shared_ptr<CompositeInstruction> in_circuit) {
  std::vector<std::shared_ptr<xacc::Instruction>> mirrorCircuit;
  auto gateProvider = xacc::getService<xacc::IRProvider>("quantum");

  const int n = in_circuit->nPhysicalBits();
  // Tracking the Pauli layer as it is commuted through
  std::vector<qcor::utils::PauliLabel> net_paulis(n,
                                                  qcor::utils::PauliLabel::I);

  // Sympletic group for Pauli gates:
  const auto srep_dict =
      qcor::utils::computeGateSymplecticRepresentations({"I", "X", "Y", "Z"});

  const auto pauliListToLayer =
      [](const std::vector<qcor::utils::PauliLabel> &in_paulis) {
        qcor::utils::CliffordGateLayer_t result;
        for (int i = 0; i < in_paulis.size(); ++i) {
          const auto pauli = in_paulis[i];
          switch (pauli) {
          case qcor::utils::PauliLabel::I:
            result.emplace_back(std::make_pair("I", i));
            break;
          case qcor::utils::PauliLabel::X:
            result.emplace_back(std::make_pair("X", i));
            break;
          case qcor::utils::PauliLabel::Y:
            result.emplace_back(std::make_pair("Y", i));
            break;
          case qcor::utils::PauliLabel::Z:
            result.emplace_back(std::make_pair("Z", i));
            break;
          default:
            __builtin_unreachable();
          }
        }
        return result;
      };
  
  // in_circuit->as_xacc()->toGraph()->write(std::cout);
  const auto decomposeU3Angle = [](xacc::InstPtr u3_gate) {
    const double theta = InstructionParameterToDouble(u3_gate->getParameter(0));
    const double phi = InstructionParameterToDouble(u3_gate->getParameter(1));
    const double lam = InstructionParameterToDouble(u3_gate->getParameter(2));
    // Convert to 3 rz angles:
    const double theta1 = lam;
    const double theta2 = theta + M_PI;
    const double theta3 = phi + 3.0 * M_PI;
    return std::make_tuple(theta1, theta2, theta3);
  };

  const auto createU3GateFromAngle = [](size_t qubit, double theta1,
                                        double theta2, double theta3) {
    auto gateProvider = xacc::getService<xacc::IRProvider>("quantum");
    return gateProvider->createInstruction(
        "U", {qubit}, {theta2 - M_PI, theta3 - 3.0 * M_PI, theta1});
  };
  const auto d = in_circuit->depth();
  for (int layer = d - 1; layer >= 0; --layer) {
    auto current_layers = getLayer(in_circuit->as_xacc(), layer);
    for (const auto &gate : current_layers) {
      // Only handle "U3" gate for now.
      // TODO: convert all single-qubit gates to U3
      if (gate->name() == "U") {
        const auto u3_angles = decomposeU3Angle(gate);
        const auto [theta1_inv, theta2_inv, theta3_inv] =
            qcor::utils::invU3Gate(u3_angles);
        const size_t qubit = gate->bits()[0];
        in_circuit->addInstruction(
            gateProvider->createInstruction("Rz", {qubit}, {theta3_inv}));
        in_circuit->addInstruction(
            gateProvider->createInstruction("Rx", {qubit}, {M_PI / 2.0}));
        in_circuit->addInstruction(
            gateProvider->createInstruction("Rz", {qubit}, {theta2_inv}));
        in_circuit->addInstruction(
            gateProvider->createInstruction("Rx", {qubit}, {M_PI / 2.0}));
        in_circuit->addInstruction(
            gateProvider->createInstruction("Rz", {qubit}, {theta1_inv}));
      }
    }
  }

  for (int layer = 0; layer < d; ++layer) {
    auto current_layers = getLayer(in_circuit->as_xacc(), layer);
    // New random Pauli layer
    const std::vector<qcor::utils::PauliLabel> new_paulis = [](int nQubits) {
      static std::random_device rd;
      static std::mt19937 gen(rd());
      static std::uniform_int_distribution<size_t> dis(
          0, qcor::utils::ALL_PAULI_OPS.size() - 1);
      std::vector<qcor::utils::PauliLabel> random_paulis;
      for (int i = 0; i < nQubits; ++i) {
        random_paulis.emplace_back(qcor::utils::ALL_PAULI_OPS[dis(gen)]);
      }

      return random_paulis;
    }(n);

    const auto new_paulis_as_layer = pauliListToLayer(new_paulis);
    const auto current_net_paulis_as_layer = pauliListToLayer(net_paulis);
    const auto new_net_paulis_reps =
        qcor::utils::computeCircuitSymplecticRepresentations(
            {new_paulis_as_layer, current_net_paulis_as_layer}, n, srep_dict);

    // Update the tracking net
    net_paulis = qcor::utils::find_pauli_labels(new_net_paulis_reps.second);
    for (const auto &gate : current_layers) {
      // Only handle "U3" gate for now.
      // TODO: convert all single-qubit gates to U3
      if (gate->name() == "U") {
        const size_t qubit = gate->bits()[0];
        const auto [theta1, theta2, theta3] = decomposeU3Angle(gate);
        // Compute the pseudo_inverse gate:
        const auto [theta1_new, theta2_new, theta3_new] =
            qcor::utils::computeRotationInPauliFrame(
                std::make_tuple(theta1, theta2, theta3), new_paulis[qubit],
                net_paulis[qubit]);
        mirrorCircuit.emplace_back(
            createU3GateFromAngle(qubit, theta1_new, theta2_new, theta3_new));
      }
    }
  }

  const auto [telp_s, telp_p] =
      qcor::utils::computeLayerSymplecticRepresentations(
          pauliListToLayer(net_paulis), n, srep_dict);
  std::vector<bool> target_bitString;
  for (int i = n; i < telp_p.size(); ++i) {
    target_bitString.emplace_back(telp_p[i] == 2);
  }

  auto mirror_comp =
      gateProvider->createComposite(in_circuit->name() + "_MIRROR");
  mirror_comp->addInstructions(mirrorCircuit);
  return std::make_pair(std::make_shared<CompositeInstruction>(mirror_comp),
                        target_bitString);
}
} // namespace qcor
