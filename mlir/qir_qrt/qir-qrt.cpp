#include "qir-qrt.hpp"

#include <alloca.h>

// #include "qcor.hpp"
#include "config_file_parser.hpp"
#include "qcor_config.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_config.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

Result ResultZero = false;
Result ResultOne = true;
// Track allocated qubits
unsigned long allocated_qbits = 0;
std::shared_ptr<xacc::AcceleratorBuffer> global_qreg;
std::shared_ptr<xacc::Accelerator> qpu;
std::string qpu_name = "qpp";
std::string qpu_config = "";
QRT_MODE mode = QRT_MODE::FTQC;
std::vector<std::unique_ptr<Array>> allocated_arrays;
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

void __quantum__rt__initialize(int argc, int8_t** argv) {
  char** casted = reinterpret_cast<char**>(argv);
  std::vector<std::string> args(casted, casted + argc);

  mode = QRT_MODE::FTQC;
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
  }
}

void __quantum__rt__set_external_qreg(qreg* q) {
  global_qreg = xacc::as_shared_ptr(q->results());
  external_qreg_provided = true;
}



Array* __quantum__rt__qubit_allocate_array(uint64_t size) {
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
    ::quantum::set_current_buffer(global_qreg.get());
  }
  // Update size.
  global_qreg.get()->setSize(allocated_qbits);

  auto raw_ptr = new_array.get();
  allocated_arrays.push_back(std::move(new_array));
  return raw_ptr;
}

int8_t* __quantum__rt__array_get_element_ptr_1d(Array* q, uint64_t idx) {
  Array &arr = *q;
  int8_t *ptr = arr[idx];
  // Don't deref the underlying type since we don't know what it points to.
  if (verbose)
    printf("[qir-qrt] Returning array element at idx=%lu.\n", idx);
  return ptr;
}

void __quantum__rt__qubit_release_array(Array* q) {
  // Note: calling qubit_release_array means the Qubits
  // are permanently deallocated.
  // Shallow references (e.g. in array slices) could become dangling if not
  // properly managed.
  // We don't resize the global buffer in this case, i.e. these
  // qubit index numbers (unique) are unused
  // => the backend won't apply any further instructions on these.
  for (std::size_t i = 0; i < allocated_arrays.size(); i++) {
    if (allocated_arrays[i].get() == q) {
      auto& array_ptr = allocated_arrays[i];
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
    }
  }
}

void __quantum__rt__finalize() {
  if (verbose) std::cout << "[qir-qrt] Running finalization routine.\n";
  if (mode == QRT_MODE::NISQ) {
    xacc::internal_compiler::execute_pass_manager();
    ::quantum::submit(global_qreg.get());
    auto counts = global_qreg->getMeasurementCounts();
    std::cout << "Observed Counts:\n";
    for (auto [bits, count] : counts) {
      std::cout << bits << " : " << count << "\n";
    }
  } else if (external_qreg_provided) {
    xacc::internal_compiler::execute_pass_manager();
    ::quantum::submit(global_qreg.get());
  }
}

bool __quantum__rt__result_equal(Result *res, Result *comp) {
  if (verbose) std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
  // std::cout << "RES = " << res << "\n";
  // std::cout << "COMP = " << comp << "\n";
  // We can do pointer comparison here.
  return res == comp;
}

void __quantum__rt__string_update_reference_count(void *str, int64_t count) {
  // TODO
  if (verbose) std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}

void __quantum__rt__result_update_reference_count(Result *, int64_t count) {
  // TODO
  if (verbose) std::cout << "CALL: " << __PRETTY_FUNCTION__ << "\n";
}
