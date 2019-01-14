#include "AMeas.hpp"
#include "XACC.hpp"

namespace xacc {
namespace quantum {
AMeas::AMeas() : GateInstruction(name()) {}

AMeas::AMeas(std::vector<int> qbs) : GateInstruction(name(), qbs) {}

const std::string AMeas::name() const {
  // Name used to construct IR instances,
  return "ameas";
}

const std::string AMeas::description() const {
  // Pulse waveform
  return "";
}

const bool AMeas::isAnalog() const { return true; }
const int AMeas::nRequiredBits() const { return 1; }

void AMeas::customVisitAction(BaseInstructionVisitor &iv) {

  // CACHING IMPLEMENTED, need to create qcor::getPulseLibrary function
  std::map<std::string, InstructionParameter> pulseLibrary =
      xacc::getCache("qcor_pulse_library.json"); //= qcor::getPulseLibrary();

  // Set the channel
  if (!options.count("t0")) {
    options.insert({"t0", InstructionParameter(64)});
  }

  if (options["pulse_id"].toString() == "null") {
    // No samples provided, but we have an angle
    // so generate them based on angle

    // GENERATE SAMPLES FROM THETA
    xacc::error("Currently invalid pulse_id (null). We do not generate pulses "
                "from angles yet.");
    //   auto theta = getParameter(0).as<double>();
    // FIXME GENERATE new_name_we_create samples
    //   options.insert({"pulse_id",
    //   InstructionParameter("new_name_we_create")});

    // qcor::appendPulse(new_name_we_create_samples);

  } else {
    // user did provide pulse id, so just make
    // sure it exists in the pulseLibrary
    auto pulseId = options["pulse_id"].as<std::string>();
    if (!pulseLibrary.count(pulseId)) {
      xacc::error("This pulse_id (" + pulseId +
                  ") is not in the pulse library cache.");
    }
  }

  // We now have valid ch, t0, and pulse_id,
  // so build up instruction json
  std::stringstream ss;
  ss << "\"ch\": \"m" << bits()[0] << "\", \"name\": \""
     << options["pulse_id"].as<std::string>()
     << "\", \"t0\":" << options["t0"].toString();
  iv.getNativeAssembly() += "{ " + ss.str() + " },";
  iv.getNativeAssembly() +=
      "{\"name\": \"acquire\", \"qubits\": ["+std::to_string(bits()[0])+"], \"memory_slot\": [0], "
      "\"register_slot\": [0], \"duration\": 512, \"kernels\": [{\"name\": "
      "\"boxcar\", \"params\": {\"start_window\": 0, \"stop_window\": 512}}], "
      "\"t0\": 64},";
}

} // namespace quantum
} // namespace xacc
