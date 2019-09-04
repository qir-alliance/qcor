#include "YPulse.hpp"
#include "xacc.hpp"

namespace xacc {
namespace quantum {
YPulse::YPulse()
    : GateInstruction(name(), std::vector<InstructionParameter>{
                                  InstructionParameter(0.0)}) {}
YPulse::YPulse(std::vector<int> qbs)
    : GateInstruction(
          name(), qbs,
          std::vector<InstructionParameter>{InstructionParameter(0.0)}) {}

const std::string YPulse::name() const {
  // Name used to construct IR instances,
  return "YPulse";
}

const std::string YPulse::description() const {
  // Pulse waveform
  return "";
}

const bool YPulse::isAnalog() const { return true; }
const int YPulse::nRequiredBits() const { return 1; }

void YPulse::customVisitAction(BaseInstructionVisitor &iv) {

  // CACHING IMPLEMENTED, need to create qcor::getPulseLibrary function
  std::map<std::string, InstructionParameter> pulseLibrary =
      xacc::getCache("qcor_pulse_library.json"); //= qcor::getPulseLibrary();

  // Set the channel
  if (!options.count("t0")) {
    options.insert({"t0", InstructionParameter(0.0)});
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
  ss << "\"ch\": \"d" << bits()[0] << "\", \"name\": \""
     << options["pulse_id"].as<std::string>()
     << "\", \"t0\":" << options["t0"].toString();
  iv.getNativeAssembly() += "{ " + ss.str() + " },";
}

} // namespace quantum
} // namespace xacc
