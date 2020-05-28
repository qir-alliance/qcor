#include "Observable.hpp"
#include "qcor.hpp"

#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"

#include "Instruction.hpp"
#include "InstructionIterator.hpp"
#include "Utils.hpp"

#include <memory>
#include <set>

#include <Eigen/Dense>

using namespace cppmicroservices;

namespace qcor {

class RBMChemistry : public ObjectiveFunction {
public:
  void initialize(std::shared_ptr<xacc::Observable> obs,
                  xacc::CompositeInstruction *qk) override {
    ObjectiveFunction::initialize(obs, qk);
    std::cout << "DW\n" << qk->toString() << "\n";

    c = 0.0;

    ham_mat_elements = obs->to_sparse_matrix();
    for (auto &x : ham_mat_elements) {
      std::cout << x.row() << ", " << x.col() << ", " << x.coeff() << "\n";
    }
  }

protected:
  int nv;
  int nh;
  std::vector<xacc::SparseTriplet> ham_mat_elements;
  Eigen::VectorXd d_vec;
  double c;

  double operator()() override {
    auto tmp_child = qalloc(qreg.size());
    for (auto i : kernel->getInstructions()) {
      std::cout << i->name() << ", " << i->isComposite() << "\n";
      if (i->isComposite() && i->name() == "rbm") {
        nv = i->getParameter(0).as<int>();
        nh = i->getParameter(1).as<int>();
        break;
      }
    }

    d_vec = Eigen::VectorXd::Zero(nv);

    std::cout << "Args; " << kernel->getArguments().size() << "\n";
    std::cout << kernel->getArguments()[0]->name << "\n";

    // Get vis_bias, hid_bias, wij, d, and c
    auto current_parameters =
        kernel->getArguments()[1]->runtimeValue.get<std::vector<double>>(
            xacc::INTERNAL_ARGUMENT_VALUE_KEY);
    std::cout << "Current Params: " << current_parameters.size() << "\n";
    int counter = 0;
    for (int i = nv + nh + nv * nh; i < current_parameters.size() - 1; i++) {
      d_vec(counter) = current_parameters[i];
      counter++;
    }

    c = current_parameters[current_parameters.size() - 1];

    std::cout << "EXECUTING\n";
    auto tmp_kernel =
        xacc::getIRProvider("quantum")->createComposite("tmp_rbm");
    xacc::InstructionIterator iter(xacc::as_shared_ptr(kernel));
    while (iter.hasNext()) {
      auto next = iter.next();
      if (!next->isComposite()) {
        tmp_kernel->addInstruction(next);
      }
    }

    // Here we have an evaluated RBM, execute it, and get its counts back
    xacc::internal_compiler::execute(tmp_child.results(), tmp_kernel.get());
    auto counts = tmp_child.counts();
    std::cout << "EXECUTED\n";
    int shots = 0;
    std::vector<std::vector<int>> states;
    std::vector<int> state_counts_vec;
    std::vector<std::string> states_str;
    for (auto &kv : counts) {
      std::cout << "COUNT: " << kv.first << ", " << kv.second << "\n";
      std::vector<int> tmp;
      std::string tmp_str = "";
      for (int i = 0; i < nv; i++) {
        tmp.push_back(kv.first[i] == '1' ? 1 : -1);
        tmp_str += kv.first[i];
      }
      state_counts_vec.push_back(kv.second);
      states.push_back(tmp);
      states_str.push_back(tmp_str);
      shots += kv.second;
    }
    
    Eigen::VectorXd state_counts(state_counts_vec.size());
    for (int i = 0; i < state_counts_vec.size(); i++) {
        std::cout << "state ocutns : " << state_counts_vec [i] << "\n";
        state_counts(i) = state_counts_vec[i] / (double) shots;
    }

    Eigen::MatrixXd sMat(states.size(), states[0].size());
    sMat.setZero();

    for (int s = 0; s < states.size(); s++) {
      for (int i = 0; i < states[0].size(); i++) {
        sMat(s, i) = (double)states[s][i];
      }
    }

    std::cout << "smat:\n" << sMat << "\nd_ved\n" << d_vec << "\n";

    // Cmopute sx
    Eigen::MatrixXd d_reshaped = Eigen::Map<Eigen::MatrixXd>(d_vec.data(), nv, 1);
    Eigen::MatrixXd ttmp = sMat * d_reshaped + c*Eigen::MatrixXd::Ones(sMat.rows(), 1) ;
    ttmp = ttmp.array().tanh();
    Eigen::VectorXd sx = Eigen::Map<Eigen::VectorXd>(ttmp.data(), sMat.rows());

    std::cout << "S(X): " << sx.transpose() << "\n";

    //   sx = np.tanh(np.dot(states, d.reshape((nv,1))) + c).reshape((1,
    //   num_states))

    //     # Compute probabilities.
    //     probs = np.multiply((sx**2).flatten(), state_counts**n)
    //     probs = probs / np.sum(probs)

    //     # Compute psi.
    //     psi = np.multiply(np.sign(sx.flatten()), np.sqrt(probs))

    Eigen::VectorXd probs = Eigen::VectorXd::Zero(state_counts.size());
    for (int i = 0; i < sx.rows(); i++) {
        std::cout << "prob: " << state_counts(i) << "\n";
      probs(i) = sx(i) * sx(i) * state_counts(i);
    }

    probs = probs / probs.sum();

    auto sign = [](auto val) { return (0.0 < val) - (val < 0.0); };

    std::cout << "SIGN: " << sign(-1) << ", " << sign (1) << "\n";
    Eigen::VectorXd psi = Eigen::VectorXd::Zero(state_counts.size());
    for (int i = 0; i < psi.rows(); i++) {
      psi(i) = sign(sx(i)) * std::sqrt(probs(i));
    }
    std::cout << "probs:\n" << probs.transpose() << "\n\n";
    std::cout << "psi\n" << psi.transpose() << "\n";

// for i in range(num_states):
//         local_energy = 0
//         x = states_str[i]
//         for j in range(num_states):
//             x1 = states_str[j]
//             if (x, x1) in ham_mat_elements:
//                 local_energy += ham_mat_elements[(x,x1)]*psi[j]
//         local_energy /= psi[i]
//         Eloc.append(local_energy)

    Eigen::VectorXd Eloc = Eigen::VectorXd::Zero(psi.rows());
    for (int i = 0; i < Eloc.rows(); i++) {
      double local_energy = 0.0;
      auto x = states_str[i];
      std::cout << "X IS " << x << "\n";
      auto x_i = std::stoi(x, 0, 2);
      for (int j = 0; j < states_str.size(); j++) {
        auto x1 = states_str[j];
        auto x1_j = std::stoi(x1, 0, 2);
        std::cout << "X1 IS " << x1 << "\n";

        auto found_triplet =
            std::find_if(ham_mat_elements.begin(), ham_mat_elements.end(),
                         [&](xacc::SparseTriplet &val) {
                           if (val.row() == x_i && val.col() == x1_j) {
                             return true;
                           }
                           return false;
                         });

        if (found_triplet != ham_mat_elements.end()) {
          local_energy += found_triplet->coeff().real() * psi[j];
        }
      }
      std::cout << "adding energy : " << local_energy << "\n";
      local_energy /= psi[i];
      Eloc(i) = local_energy;
    }

    double energy = probs.dot(Eloc);


    // dA = n * states.T / 2
    // weights = np.asarray(wij).reshape((nv,nh))
    // theta = np.dot(weights.transpose(), states.T) + b.reshape((nh,1))
    // dB = n * np.tanh(theta) / 2
    // dW = (states.T).reshape((nv,1,num_states)) * dB.reshape((1,nh,num_states)) 
    // oneoversx = np.clip(1/sx, -10**10, 10**10)
    // dc = oneoversx - sx
    // dd = states.T * dc
    // dP = np.concatenate([dA,dB,dW.reshape(nv*nh, num_states),dd,dc])
    
    // # Compute S and F.
    // E_dP = np.dot(dP, probs.reshape((num_states, 1)))
    // dP_conj = np.conjugate(dP)
    // E_dP_conj = np.dot(dP_conj, probs.reshape((num_states, 1)))
    
    // dP_with_probs = np.multiply(dP, probs.reshape((1,num_states)))
    // S = np.dot(dP_conj, dP_with_probs.T) - np.dot(E_dP_conj, E_dP.T)
    // F = np.dot(dP_conj, (Eloc*probs).reshape(num_states,1)) - energy*E_dP_conj

    // # Solve S * \delta p = F
    // update, info = linalg.bicgstab(S + epsilon * np.identity(S.shape[0]), F, x0=update)


    std::cout << "Energy: " << energy << "\n";
    // exit(0);
    qreg.addChild(tmp_child);
    return energy;
  }

public:
  const std::string name() const override { return "rbm-chemistry"; }
  const std::string description() const override { return ""; }
};

} // namespace qcor

namespace {

/**
 */
class US_ABI_LOCAL RBMChemObjectiveActivator : public BundleActivator {

public:
  RBMChemObjectiveActivator() {}

  /**
   */
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::RBMChemistry>();
    context.RegisterService<qcor::ObjectiveFunction>(xt);
  }

  /**
   */
  void Stop(BundleContext /*context*/) {}
};

} // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(RBMChemObjectiveActivator)
