#include "qcor.hpp"
#include "qpu_handler.hpp"

#include "AcceleratorBuffer.hpp"
#include "IRProvider.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

#include "FermionOperator.hpp"
#include "PauliOperator.hpp"

#include <regex>

using namespace xacc;

namespace qcor {
std::map<std::string, InstructionParameter> runtimeMap = {};

void Initialize(int argc, char **argv) {
  std::vector<const char *> tmp(argv, argv + argc);
  std::vector<std::string> newargv;
  for (auto &t : tmp)
    newargv.push_back(std::string(t));
  Initialize(newargv);
}
void Initialize(std::vector<std::string> argv) {
  argv.push_back("--logger-name");
  argv.push_back("qcor");
  xacc::Initialize(argv);
}

const std::string persistCompiledCircuit(std::shared_ptr<Function> function,
                                         std::shared_ptr<Accelerator> acc) {
  srand(time(NULL));
  std::function<char()> randChar = []() -> char {
    const char charset[] = "0123456789"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };

  auto generateRandomString = [&](const int length = 10) -> const std::string {
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randChar);
    return str;
  };

  std::string file_name;
  if (xacc::optionExists("qcor-compiled-filename")) {
    file_name = xacc::getOption("qcor-compiled-filename");
  } else {

    file_name = generateRandomString();
    // std::cout << "Generating random string " << file_name << "\n";
  }

  std::stringstream ss;
  function->persist(ss);
  auto persistedFunction = ss.str();
//   std::cout << "PERSISTED\n" << persistedFunction << "\n";
  //   xacc::getCompiler("xacc-py")->translate("", function);
  //   persistedFunction = persistedFunction.substr(7,
  //   persistedFunction.length());
  xacc::appendCache(file_name, "compiled",
                    InstructionParameter(persistedFunction), ".qcor_cache");

  if (acc) {
    xacc::appendCache(file_name, "accelerator",
                      InstructionParameter(acc->name()), ".qcor_cache");
  } else {
    xacc::appendCache(file_name, "accelerator",
                      InstructionParameter("default-sim"), ".qcor_cache");
  }

  if (function->hasIRGenerators()) {
    xacc::appendCache(file_name, "requires-jit", InstructionParameter("true"),
                      ".qcor_cache");
  } else {
    xacc::appendCache(file_name, "requires-jit", InstructionParameter("false"),
                      ".qcor_cache");
  }

  bool isDw = false;
  for (auto& inst : function->getInstructions()) {
      if (inst->name() == "dw-qmi" || inst->name() == "anneal") {
          isDw = true;
          break;
      }
  }

  if (isDw) {
    xacc::appendCache(file_name, "ir-type", InstructionParameter("anneal"),
                      ".qcor_cache");
  } else {
    xacc::appendCache(file_name, "ir-type", InstructionParameter("gate"),
                      ".qcor_cache");
  }
  return file_name;
}

std::shared_ptr<Function> loadCompiledCircuit(const std::string &fileName) {
  //   std::cout << "Loading Circuit " << fileName << "\n";
  auto cache = xacc::getCache(fileName, ".qcor_cache");
  if (!cache.count("compiled")) {
    xacc::error("Invalid quantum compilation cache.");
  }

  std::shared_ptr<Accelerator> targetAccelerator;
  if (cache["accelerator"] == "default-sim") {
    // First, if this is compiled for simulation,
    // let users set the simulator at the command line
    // if they didnt, check for tnqvm, then local-ibm
    if (xacc::optionExists("accelerator")) {
      targetAccelerator = xacc::getAccelerator();
    } else if (xacc::hasAccelerator("tnqvm")) {
      targetAccelerator = xacc::getAccelerator("tnqvm");
    } else if (xacc::hasAccelerator("local-ibm")) {
      targetAccelerator = xacc::getAccelerator("local-ibm");
    }
  } else {
    auto accStr = cache["accelerator"].as<std::string>();
    if ((accStr == "tnqvm" || accStr == "local-ibm") &&
        xacc::optionExists("accelerator")) {
      targetAccelerator = xacc::getAccelerator();
    } else {
      targetAccelerator =
          xacc::getAccelerator(cache["accelerator"].as<std::string>());
    }
  }

  // If for some reason we still dont have an
  // accelerator, force them to specify at command line
  if (!targetAccelerator) {
    targetAccelerator = xacc::getAccelerator();
  }

  xacc::setAccelerator(targetAccelerator->name());

  auto type = cache["ir-type"].as<std::string>();
  auto compiled = cache["compiled"].as<std::string>();
  auto loaded =
      xacc::getService<IRProvider>("quantum")->createFunction("loaded", {}, {InstructionParameter(type)});
  std::istringstream iss(compiled);
  loaded->load(iss);

  //   std::cout << "Lodaded: " << loaded->toString() << "\n";
  if (cache["requires-jit"].as<std::string>() == "true") {
    auto runtimeMap = getRuntimeMap();

    // for (auto& kv : runtimeMap) {
    // std::cout << "Runtime: " << kv.first << ", " << kv.second.toString() <<
    // "\n";
    // }
    loaded->expandIRGenerators(runtimeMap);

    // Kick off quantum compilation
    auto qcor = xacc::getCompiler("qcor");
    loaded = qcor->compile(loaded, targetAccelerator);
    // std::cout << "NPARAMS: " << loaded->nParameters() << "\n";
  }

  //   std::cout<< "Loaded IR:\n" << loaded->toString() <<"\n";
  return loaded;
}

void storeRuntimeVariable(const std::string name, InstructionParameter param) {
  //   std::cout << "Storing Runtime Variable " << name << ", " <<
  //   param.toString() << "\n";
  runtimeMap.insert({name, param});
}

std::map<std::string, InstructionParameter> getRuntimeMap() {
  return runtimeMap;
}

std::future<std::shared_ptr<AcceleratorBuffer>>
submit(HandlerLambda &&totalJob) {
  // Create the QPU Handler to pass to the given
  // Handler HandlerLambda
  return std::async(std::launch::async, [=]() { // bug must be by value...
    qpu_handler handler;
    totalJob(handler);
    return handler.getResults();
  });
}

std::shared_ptr<Optimizer> getOptimizer(const std::string &name) {
  return xacc::getService<qcor::Optimizer>(name);
}
std::shared_ptr<Optimizer>
getOptimizer(const std::string &name,
             std::map<std::string, InstructionParameter> &&options) {
  auto opt = getOptimizer(name);
  opt->setOptions(options);
  return opt;
}

std::shared_ptr<Observable> getObservable(const std::string &type,
                                          const std::string &representation) {
  using namespace xacc::quantum;
  if (type == "pauli") {
    return representation.empty()
               ? std::make_shared<PauliOperator>()
               : std::make_shared<PauliOperator>(representation);
  } else if (type == "fermion") {
    return representation.empty()
               ? std::make_shared<FermionOperator>()
               : std::make_shared<FermionOperator>(representation);
  } else {
    xacc::error("Invalid observable type");
    return std::make_shared<PauliOperator>();
  }
}

std::shared_ptr<Observable> getObservable() {
  return getObservable("pauli", "");
}
std::shared_ptr<Observable> getObservable(const std::string &representation) {
  return getObservable("pauli", representation);
}

std::shared_ptr<Observable>
getObservable(const std::string &type,
              std::map<std::string, InstructionParameter> &&options) {
  auto observable = xacc::getService<Observable>(type);
  observable->fromOptions(options);
  return observable;
}
std::shared_ptr<algorithm::Algorithm> getAlgorithm(const std::string name) {
  return xacc::getService<qcor::algorithm::Algorithm>(name);
}

} // namespace qcor