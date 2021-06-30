#include "qcor_qsim.hpp"
#include "xacc_service.hpp"
#include "IRProvider.hpp"

namespace qcor {
namespace QuaSiMo {
bool CostFunctionEvaluator::initialize(Operator *observable,
                                       const HeterogeneousMap &params) {
  target_operator = observable;
  hyperParams = params;
  return target_operator != nullptr;
}

void executePassManager(
    std::vector<std::shared_ptr<CompositeInstruction>> evalKernels) {
  for (auto &subKernel : evalKernels) {
    execute_pass_manager(subKernel);
  }
}

QuantumSimulationModel
ModelFactory::createModel(Operator *obs, TdObservable td_ham,
                          const HeterogeneousMap &params) {
  QuantumSimulationModel model;
  model.observable = obs;
  model.hamiltonian = td_ham;
  return model;
}

QuantumSimulationModel
ModelFactory::createModel(Operator *obs, const HeterogeneousMap &params) {
  QuantumSimulationModel model;
  model.observable = obs;
  model.hamiltonian = [&](double t) {
    return *(static_cast<Operator *>(obs));
  };
  return model;
}

QuantumSimulationModel
ModelFactory::createModel(ModelType type, const HeterogeneousMap &params) {
  if (type == ModelType::Heisenberg) {
    // Static internal HeisenbergModel struct
    // (keep alive to form time-depenedent Hamiltonian operator)
    static std::unique_ptr<HeisenbergModel> hs_model;
    hs_model = std::make_unique<HeisenbergModel>();
    hs_model->fromDict(params);
    if (!hs_model->validateModel()) {
      qcor::error(
          "Failed to validate the input parameters for the Heisenberg model.");
    }

    QuantumSimulationModel model;

    // Handle custom observable:
    qcor::Operator *model_observable = nullptr;
    // Get QCOR Pauli Op first (if any)
    if (params.keyExists<qcor::Operator>("observable")) {
      static auto cached_pauli = params.get<qcor::Operator>("observable");
      model_observable = &cached_pauli;
    }
    // Generic qcor::Operator
    else if (params.pointerLikeExists<qcor::Operator>("observable")) {
      model_observable = params.getPointerLike<qcor::Operator>("observable");
    }
    else {
      const std::vector<std::string> SUPPORTED_OBS{"average_magnetization",
                                                   "staggered_magnetization"};
      std::string observable_type = "average_magnetization";
      if (params.stringExists("observable")) {
        observable_type = params.getString("observable");
      }

      if (!xacc::container::contains(SUPPORTED_OBS, observable_type)) {
        qcor::error("Unknown observable type: " + observable_type);
      }

      auto observable = new qcor::Operator("pauli", "0.0");
      if (observable_type == "average_magnetization") {
        for (int i = 0; i < hs_model->num_spins; ++i) {
          (*observable) += ((1.0 / hs_model->num_spins) * Z(i));
        }
      }
      else if (observable_type == "staggered_magnetization") {
        double coeff = 1.0;
        for (int i = 0; i < hs_model->num_spins; ++i) {
          (*observable) += ((coeff / hs_model->num_spins) * Z(i));
          // Alternate the sign
          coeff = -coeff;
        }
      }

      model_observable = observable;
    }

    assert(model_observable);
    model.observable = model_observable;
    QuaSiMo::TdObservable H = [&](double t) {
      qcor::Operator tdOp("pauli", "0.0");
      for (int i = 0; i < hs_model->num_spins - 1; ++i) {
        if (hs_model->Jx != 0.0) {
          tdOp += ((hs_model->Jx / hs_model->H_BAR) * (X(i) * X(i + 1)));
        }
        if (hs_model->Jy != 0.0) {
          tdOp += ((hs_model->Jy / hs_model->H_BAR) * (Y(i) * Y(i + 1)));
        }
        if (hs_model->Jz != 0.0) {
          tdOp += ((hs_model->Jz / hs_model->H_BAR) * (Z(i) * Z(i + 1)));
        }
      }

      std::function<double(double)> time_dep_func;
      if (hs_model->time_func) {
        time_dep_func = hs_model->time_func;
      } else {
        time_dep_func = [&](double time) { return std::cos(hs_model->freq * time); };
      }
      if (hs_model->h_ext != 0.0) {
        for (int i = 0; i < hs_model->num_spins; ++i) {
          if (hs_model->ext_dir == "X") {
            tdOp +=
                ((hs_model->h_ext / hs_model->H_BAR) * time_dep_func(t) * X(i));
          } else if (hs_model->ext_dir == "Y") {
            tdOp +=
                ((hs_model->h_ext / hs_model->H_BAR) * time_dep_func(t) * Y(i));
          } else {
            tdOp +=
                ((hs_model->h_ext / hs_model->H_BAR) * time_dep_func(t) * Z(i));
          }
        }
      }

      return tdOp;
    };
    
    model.hamiltonian = H;
    // Non-zero initial spin state:
    if (std::find(hs_model->initial_spins.begin(), hs_model->initial_spins.end(),
                  1) != hs_model->initial_spins.end()) {
      auto initialSpinPrep = qcor::__internal__::create_composite("InitialSpin");
      auto provider = qcor::__internal__::get_provider();
      for (int i = 0; i < hs_model->initial_spins.size(); ++i) {
        if (hs_model->initial_spins[i] == 1) {
          initialSpinPrep->addInstruction(provider->createInstruction("X", i));
        }
      }
      model.user_defined_ansatz = std::make_shared<KernelFunctor>(initialSpinPrep);
    }

    return model;
  } else {
    qcor::error("Unknown model type.");
    __builtin_unreachable();
  }
}

QuantumSimulationModel
ModelFactory::createModel(const std::string &format, const std::string &data,
                          const HeterogeneousMap &params) {
  QuantumSimulationModel model;
  // TODO:
  return model;
}

std::shared_ptr<QuantumSimulationWorkflow>
getWorkflow(const std::string &name, const HeterogeneousMap &init_params) {
  auto qsim_workflow = xacc::getService<QuantumSimulationWorkflow>(name);
  if (qsim_workflow && qsim_workflow->initialize(init_params)) {
    return qsim_workflow;
  }
  // ERROR: unknown workflow or invalid initialization options.
  return nullptr;
}

std::shared_ptr<CostFunctionEvaluator>
getObjEvaluator(Operator *observable, const std::string &name,
                const HeterogeneousMap &init_params) {
  auto evaluator = xacc::getService<CostFunctionEvaluator>(name);
  if (evaluator && evaluator->initialize(observable, init_params)) {
    return evaluator;
  }
  // ERROR: unknown CostFunctionEvaluator or invalid initialization options.
  return nullptr;
}

void ModelFactory::HeisenbergModel::fromDict(const HeterogeneousMap &params) {
  const auto getKeyIfExists = [&params](auto &modelVar,
                                        const std::string &keyName) {
    using ValType = typename std::remove_reference_t<decltype(modelVar)>;
    if (params.keyExists<ValType>(keyName)) {
      modelVar = params.get<ValType>(keyName);
    }
  };

  // Handle exact types (int, double)
  getKeyIfExists(Jx, "Jx");
  getKeyIfExists(Jy, "Jy");
  getKeyIfExists(Jz, "Jz");
  getKeyIfExists(h_ext, "h_ext");
  getKeyIfExists(H_BAR, "H_BAR");
  getKeyIfExists(num_spins, "num_spins");
  getKeyIfExists(initial_spins, "initial_spins");
  getKeyIfExists(time_func, "time_func");
  getKeyIfExists(freq, "freq");
  // Handle string-like
  if (params.stringExists("ext_dir")) {
    ext_dir = params.getString("ext_dir");
  }
}
} // namespace QuaSiMo
} // namespace qcor