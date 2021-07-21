#include "qir-qrt.hpp"

#include <alloca.h>
#include <regex>

#include "config_file_parser.hpp"
#include "qcor_config.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_config.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"
static const Result ResultZeroVal = false;
static const Result ResultOneVal = true;
// Define these global pointer constants
Result *ResultZero = const_cast<Result *>(&ResultZeroVal);
Result *ResultOne = const_cast<Result *>(&ResultOneVal);
// Track allocated qubits
unsigned long allocated_qbits = 0;
std::shared_ptr<xacc::AcceleratorBuffer> global_qreg;
std::shared_ptr<xacc::Accelerator> qpu;
std::string qpu_name = "qpp";
std::string qpu_config = "";
QRT_MODE mode = QRT_MODE::FTQC;
std::vector<std::unique_ptr<Array>> allocated_arrays;
// Map of single-qubit allocations,
// i.e. arrays of size 1.
std::unordered_map<uint64_t, Array *> single_qubit_arrays;
std::stack<std::shared_ptr<xacc::CompositeInstruction>> internal_xacc_ir;
std::stack<std::shared_ptr<::quantum::QuantumRuntime>> internal_runtimes;

int shots = 0;
bool verbose = false;
bool external_qreg_provided = false;

bool initialized = false;

void print_help() {
  std::cout << "QCOR QIR Runtime Help Menu\n\n";
  std::cout << "optional arguments:\n";
  std::cout << "  -qpu QPUNAME[:BACKEND] | example -qpu ibm:ibmq_vigo, -qpu "
               "aer:ibmq_vigo\n";
  std::cout << "  -qpu-config config_file.ini | example: -qpu ibm:ibmq_vigo "
               "-qpu-config ibm_config.ini\n";
  std::cout << "  -qrt QRT_MODE (can be nisq or ftqc) | example -qrt nisq\n";
  std::cout << "  -shots NUMSHOTS (number of shots to use in nisq run)\n";
  std::cout << "  -opt LEVEL | example -opt 1\n";
  std::cout
      << "  -print-opt-stats (turn on printout of optimization statistics) \n";
  std::cout << "  -v,-verbose,--verbose (run with printouts)\n";
  std::cout << "  -xacc-verbose (turn on extra xacc verbose print-out)\n\n";
  exit(0);
}

namespace qcor {
void initialize() { initialize(std::vector<std::string>{}); }

void initialize(std::vector<std::string> argv) {
  std::vector<char *> cstrs;
  argv.insert(argv.begin(), "appExec");
  for (auto &s : argv) {
    cstrs.push_back(&s.front());
  }
  initialize(argv.size(), cstrs.data());
}

void initialize(int argc, char **argv) {
  __quantum__rt__initialize(argc, reinterpret_cast<int8_t **>(argv));
}
qcor::qreg qalloc(const uint64_t size) {
  if (!initialized) initialize();
  return qcor::qreg(size);
}
}  // namespace qcor
std::vector<std::string> config_tracker;

void __quantum__rt__set_config_parameter(int8_t *key, int8_t *value) {
  std::string casted_key (reinterpret_cast<char *>(key));
  std::string casted_value (reinterpret_cast<char *>(value));

  if (casted_key == "qpu") {
    qpu_name = casted_value;
  } else if (casted_key == "qrt") {
    mode = casted_value == "nisq" ? QRT_MODE::NISQ : QRT_MODE::FTQC;
  } else if (casted_key == "shots") {
    shots = std::stoi(casted_value);
  }

}

void __quantum__rt__initialize(int argc, int8_t **argv) {
  char **casted = reinterpret_cast<char **>(argv);
  std::vector<std::string> args(casted, casted + argc);
  // mode = QRT_MODE::FTQC;
  for (size_t i = 0; i < args.size(); i++) {
    auto arg = args[i];
    if (arg == "-qpu") {
      qpu_name = args[i + 1];
    } else if (arg == "-qpu-config") {
      qpu_config = args[i + 1];
    } else if (arg == "-qrt") {
      mode = args[i + 1] == "nisq" ? QRT_MODE::NISQ : QRT_MODE::FTQC;
    } else if (arg == "-shots") {
      shots = std::stoi(args[i + 1]);
    } else if (arg == "-xacc-verbose") {
      verbose = true;
      xacc::set_verbose(true);
    } else if (arg == "-v") {
      verbose = true;
    } else if (arg == "-verbose") {
      verbose = true;
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help") {
      print_help();
    } else if (arg == "-h") {
      print_help();
    } else if (arg == "-opt") {
      xacc::internal_compiler::__opt_level = std::stoi(args[i + 1]);
    } else if (arg == "-print-opt-stats") {
      xacc::internal_compiler::__print_opt_stats = true;
    }
  }

  initialize();
}

void initialize() {
  if (!initialized) {
    if (verbose) printf("[qir-qrt] Initializing FTQC runtime...\n");
    // qcor::set_verbose(true);
    xacc::internal_compiler::__qrt_env = "ftqc";

    // if XACC_INSTALL_DIR != XACC_ROOT
    // then we need to pass --xacc-root-path XACC_ROOT
    //
    // Example - we are on Rigetti QCS and can't install via sudo
    // so we dpkg -x xacc to a user directory, but deb package
    // expects to be extracted to /usr/local/xacc, and xacc_config.hpp
    // points to that /usr/local/xacc. Therefore ServiceRegistry fails
    // to load plugins and libs, unless we change rootPath.
    std::string xacc_config_install_dir = std::string(XACC_INSTALL_DIR);
    std::string qcor_config_xacc_root = std::string(XACC_ROOT);
    if (xacc_config_install_dir != qcor_config_xacc_root) {
      std::vector<std::string> cmd_line{"--xacc-root-path",
                                        qcor_config_xacc_root};
      xacc::Initialize(cmd_line);
    } else {
      xacc::Initialize();
    }

    if (verbose)
      std::cout << "[qir-qrt] Running on " << qpu_name << " backend.\n";
    std::shared_ptr<xacc::Accelerator> qpu;

    xacc::HeterogeneousMap qpu_config_map;
    if (!qpu_config.empty()) {
      auto parser = xacc::getService<xacc::ConfigFileParsingUtil>("ini");
      qpu_config_map = parser->parse(qpu_config);
    }

    if (!qpu_config_map.keyExists<int>("shots") && shots > 0 &&
        mode == QRT_MODE::NISQ) {
      if (verbose)
        printf("Automatically setting shots for nisq mode execution to %d\n",
               shots);
      qpu_config_map.insert("shots", shots);
    }

    if (mode == QRT_MODE::NISQ) {
      xacc::internal_compiler::__qrt_env = "nisq";
      qpu = xacc::getAccelerator(qpu_name, qpu_config_map);
    } else {
      qpu = xacc::getAccelerator(qpu_name);
    }

    xacc::internal_compiler::qpu = qpu;
    ::quantum::qrt_impl = xacc::getService<::quantum::QuantumRuntime>(
        xacc::internal_compiler::__qrt_env);
    ::quantum::qrt_impl->initialize("empty");
    initialized = true;

    // Save the original runtime by pushing it on
    // the stack, we'll always at least have this one
    internal_runtimes.push(::quantum::qrt_impl);
  }
}

void __quantum__rt__set_external_qreg(qreg *q) {
  global_qreg = xacc::as_shared_ptr(q->results());
  external_qreg_provided = true;
}

Array *__quantum__rt__qubit_allocate_array(uint64_t size) {
  if (verbose) printf("[qir-qrt] Allocating qubit array of size %lu.\n", size);

  auto new_array = std::make_unique<Array>(size);
  for (uint64_t i = 0; i < size; i++) {
    auto qubit = Qubit::allocate();
    int8_t *arrayPtr = (*new_array)[i];
    // Sequence: Cast to arrayPtr to Qubit**
    auto qubitPtr = reinterpret_cast<Qubit **>(arrayPtr);
    // Then save the qubit *pointer* to the location.
    *qubitPtr = qubit;
  }

  allocated_qbits += size;
  if (!global_qreg) {
    global_qreg = std::make_shared<xacc::AcceleratorBuffer>(size);
    global_qreg->setName("q");
    ::quantum::set_current_buffer(global_qreg.get());
  }
  // Update size.
  global_qreg.get()->setSize(allocated_qbits);

  auto raw_ptr = new_array.get();
  allocated_arrays.push_back(std::move(new_array));
  return raw_ptr;
}

Qubit *__quantum__rt__qubit_allocate() {
  auto qArray = __quantum__rt__qubit_allocate_array(1);
  int8_t *arrayPtr = (*qArray)[0];
  Qubit *qubitPtr = *(reinterpret_cast<Qubit **>(arrayPtr));
  // We track the single-qubit array that holds this qubit.
  single_qubit_arrays[qubitPtr->id] = qArray;
  return qubitPtr;
}

void __quantum__rt__qubit_release(Qubit *q) {
  if (q == nullptr) {
    return;
  }
  if (single_qubit_arrays.find(q->id) != single_qubit_arrays.end()) {
    __quantum__rt__qubit_release_array(single_qubit_arrays[q->id]);
    single_qubit_arrays.erase(q->id);
  } else {
    throw "Illegal release of a qubit.";
  }
}

void __quantum__rt__start_pow_u_region() {
  // Create a new NISQ based runtime so that we can
  // queue up instructions and get them as a CompositeInstruction
  auto tmp_runtime = xacc::getService<::quantum::QuantumRuntime>("nisq");
  tmp_runtime->initialize("empty");

  // Add the new tmp runtime to the stack
  internal_runtimes.push(tmp_runtime);

  // Set the current runtime to the new tmp one
  ::quantum::qrt_impl = tmp_runtime;

  // Now all subsequent qrt calls will queue to our
  // temp created runtime, until we hit end_pow_u_region
  return;
}

void __quantum__rt__end_pow_u_region(int64_t power) {
  // Get the temp runtime created by start_pow_u_region.
  auto runtime = internal_runtimes.top();
  // Get the program we built up
  auto program = runtime->get_current_program();
  // Remove the tmp runtime
  internal_runtimes.pop();

  // Set the quantum runtime to the one before this
  // temp one we just popped off
  ::quantum::qrt_impl = internal_runtimes.top();

  // Now apply the U region to the given power, U^power
  for (int64_t i = 0; i < power; i++) {
    for (auto inst : program->getInstructions()) {
      ::quantum::qrt_impl->general_instruction(inst);
    }
  }

  return;
}

void __quantum__rt__start_adj_u_region() {
  // Create a new NISQ based runtime so that we can
  // queue up instructions and get them as a CompositeInstruction
  auto tmp_runtime = xacc::getService<::quantum::QuantumRuntime>("nisq");
  tmp_runtime->initialize("empty");

  // Add the new tmp runtime to the stack
  internal_runtimes.push(tmp_runtime);

  // Set the current runtime to the new tmp one
  ::quantum::qrt_impl = tmp_runtime;

  // Now all subsequent qrt calls will queue to our
  // temp created runtime, until we hit end_adj_u_region
  return;
}

void __quantum__rt__end_adj_u_region() {
  // Get the temp runtime created by start_adj_u_region.
  auto runtime = internal_runtimes.top();
  // Get the program we built up
  auto program = runtime->get_current_program();
  // Remove the tmp runtime
  internal_runtimes.pop();

  // Set the quantum runtime to the one before this
  // temp one we just popped off
  ::quantum::qrt_impl = internal_runtimes.top();

  // get the instructions
  auto instructions = program->getInstructions();

  // Assert that we don't have measurement
  if (!std::all_of(
          instructions.cbegin(), instructions.cend(),
          [](const auto &inst) { return inst->name() != "Measure"; })) {
    xacc::error(
        "Unable to create Adjoint for kernels that have Measure operations.");
  }

  auto provider = xacc::getIRProvider("quantum");
  for (std::size_t i = 0; i < instructions.size(); i++) {
    auto inst = program->getInstruction(i);
    // Parametric gates:
    if (inst->name() == "Rx" || inst->name() == "Ry" || inst->name() == "Rz" ||
        inst->name() == "CPHASE" || inst->name() == "U1" ||
        inst->name() == "CRZ") {
      inst->setParameter(0, -inst->getParameter(0).template as<double>());
    }
    // Handle U3
    if (inst->name() == "U") {
      inst->setParameter(0, -inst->getParameter(0).template as<double>());
      inst->setParameter(1, -inst->getParameter(1).template as<double>());
      inst->setParameter(2, -inst->getParameter(2).template as<double>());

    }
    // Handles T and S gates, etc... => T -> Tdg
    else if (inst->name() == "T") {
      auto tdg = provider->createInstruction("Tdg", inst->bits());
      program->replaceInstruction(i, tdg);
    } else if (inst->name() == "S") {
      auto sdg = provider->createInstruction("Sdg", inst->bits());
      program->replaceInstruction(i, sdg);
    }
  }

  // We update/replace instructions in the derived.parent_kernel composite,
  // hence collecting these new instructions and reversing the sequence.
  auto new_instructions = program->getInstructions();
  std::reverse(new_instructions.begin(), new_instructions.end());

  for (auto inst : new_instructions) {
    ::quantum::qrt_impl->general_instruction(inst);
  }

  return;
}


void __quantum__rt__mark_compute() {::quantum::qrt_impl->__begin_mark_segment_as_compute();}
void __quantum__rt__unmark_compute() {::quantum::qrt_impl->__end_mark_segment_as_compute();}

void __quantum__rt__start_ctrl_u_region() {
  // Cache the current runtime into the stack if not already.
  if (internal_runtimes.empty()) {
    internal_runtimes.push(::quantum::qrt_impl);
  }
  // Create a new NISQ based runtime so that we can
  // queue up instructions and get them as a CompositeInstruction
  auto tmp_runtime = xacc::getService<::quantum::QuantumRuntime>("nisq");
  tmp_runtime->initialize("empty");

  // Add the new tmp runtime to the stack
  internal_runtimes.push(tmp_runtime);

  // Set the current runtime to the new tmp one
  ::quantum::qrt_impl = tmp_runtime;

  // Now all subsequent qrt calls will queue to our
  // temp created runtime, until we hit end_ctrl_u_region
  return;
}

void __quantum__rt__end_ctrl_u_region(Qubit *ctrl_qbit) {
  // Get the temp runtime created by start_adj_u_region.
  auto runtime = internal_runtimes.top();
  // Get the program we built up
  auto program = std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      runtime->get_current_program()->get_as_opaque());
  // Remove the tmp runtime
  internal_runtimes.pop();

  // Set the quantum runtime to the one before this
  // temp one we just popped off
  ::quantum::qrt_impl = internal_runtimes.top();

  int ctrlIdx = ctrl_qbit->id;
  auto ctrlKernel = std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      xacc::getService<xacc::Instruction>("C-U"));
  ctrlKernel->expand({
      std::make_pair("U", program),
      std::make_pair("control-idx", ctrlIdx),
  });

  // std::cout << "Running Ctrl on " << ctrlIdx << ":\n" <<
  // ctrlKernel->toString() << "\n";
  for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
    ::quantum::qrt_impl->general_instruction(
        ctrlKernel->getInstruction(instId));
  }
  return;
}

void __quantum__rt__end_multi_ctrl_u_region(
    const std::vector<Qubit *> &ctrl_qubits) {
  // Get the temp runtime created by start_adj_u_region.
  auto runtime = internal_runtimes.top();
  // Get the program we built up
  auto program = std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      runtime->get_current_program()->get_as_opaque());
  // Remove the tmp runtime
  internal_runtimes.pop();

  // Set the quantum runtime to the one before this
  // temp one we just popped off
  ::quantum::qrt_impl = internal_runtimes.top();

  std::vector<int> ctrl_bits;
  for (auto &qb : ctrl_qubits) {
    ctrl_bits.emplace_back(qb->id);
  }

  auto ctrlKernel = std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      xacc::getService<xacc::Instruction>("C-U"));
  ctrlKernel->expand({
      std::make_pair("U", program),
      std::make_pair("control-idx", ctrl_bits),
  });

  // std::cout << "Running Ctrl on ";
  // for (const auto &bit : ctrl_bits) {
  //   std::cout << bit << " ";
  // }
  // std::cout << ":\n" << ctrlKernel->toString() << "\n";
  for (int instId = 0; instId < ctrlKernel->nInstructions(); ++instId) {
    ::quantum::qrt_impl->general_instruction(
        ctrlKernel->getInstruction(instId));
  }
  return;
}

int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t idx) {
  Array &arr = *q;
  int8_t *ptr = arr[idx];
  // Don't deref the underlying type since we don't know what it points to.
  if (verbose) printf("[qir-qrt] Returning array element at idx=%lu.\n", idx);
  return ptr;
}

void __quantum__rt__qubit_release_array(Array *q) {
  // Note: calling qubit_release_array means the Qubits
  // are permanently deallocated.
  // Shallow references (e.g. in array slices) could become dangling if not
  // properly managed.
  // We don't resize the global buffer in this case, i.e. these
  // qubit index numbers (unique) are unused
  // => the backend won't apply any further instructions on these.
  for (std::size_t i = 0; i < allocated_arrays.size(); i++) {
    if (allocated_arrays[i] && allocated_arrays[i].get() == q) {
      auto &array_ptr = allocated_arrays[i];
      auto array_size = array_ptr->size();
      if (verbose && mode == QRT_MODE::FTQC)
        printf("[qir-qrt] deallocating the qubit array of size %lu\n",
               array_size);
      for (int k = 0; k < array_size; k++) {
        int8_t *arrayPtr = (*array_ptr)[k];
        Qubit *qubitPtr = *(reinterpret_cast<Qubit **>(arrayPtr));
        delete qubitPtr;
      }
      array_ptr->clear();
      array_ptr.reset();
    }
  }

  // If all the runtime arrays have been cleared/dealocated,
  // clean up the entire global register.
  // This is to handle **multiple** calls into an FTQC kernels (Q#/OpenQASM3).
  // At the end of the execution, all registers have been deallocated.
  if (std::all_of(allocated_arrays.begin(), allocated_arrays.end(),
                  [](auto &array_ptr) { return array_ptr == nullptr; }) &&
      mode == QRT_MODE::FTQC) {
    if (verbose) {
      std::cout << "Reset global buffer.\n";
    }
    allocated_arrays.clear();
    global_qreg.reset();
    allocated_qbits = 0;
    Qubit::reset_counter();
  }
}

void __quantum__rt__finalize() {
  if (verbose) std::cout << "[qir-qrt] Running finalization routine.\n";
  if (mode == QRT_MODE::NISQ) {
    xacc::internal_compiler::execute_pass_manager();
    ::quantum::submit(global_qreg.get());
    auto counts = global_qreg->getMeasurementCounts();
    if (!counts.empty()) {
      std::cout << "Observed Counts:\n";
      for (auto [bits, count] : counts) {
        std::cout << bits << " : " << count << "\n";
      }
    } else {
      std::cout << "Result Buffer:\n";
      global_qreg->print();
    }
  } else if (external_qreg_provided) {
    xacc::internal_compiler::execute_pass_manager();
    ::quantum::submit(global_qreg.get());
  }
}

bool __quantum__rt__result_equal(Result *res, Result *comp) {
  if (verbose) std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  if (mode == QRT_MODE::NISQ) {
    throw std::runtime_error(
        "Comparing Measure results in NISQ mode is prohibited.");
  }

  // std::cout << "RES = " << res << "\n";
  // std::cout << "COMP = " << comp << "\n";
  if (res && comp) {
    // Do pointer comparison here then fallback by deref the result.
    return (res == comp) || (*res) == (*comp);
  }
  return false;
}

Result *__quantum__rt__result_get_one() { return ResultOne; }
Result *__quantum__rt__result_get_zero() { return ResultZero; }

void __quantum__rt__result_update_reference_count(Result *, int32_t count) {
  // TODO
  if (verbose) std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}

void __quantum__rt__fail(QirString *str) {
  throw std::runtime_error(str->m_str);
}

void __quantum__rt__message(QirString *str) {
  std::cout << "[QIR Message] " << str->m_str << "\n";
}