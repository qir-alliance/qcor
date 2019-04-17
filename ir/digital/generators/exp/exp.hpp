#ifndef QCOR_INSTRUCTIONS_EXP_HPP_
#define QCOR_INSTRUCTIONS_EXP_HPP_

#include "IRGenerator.hpp"

namespace xacc {
    class AcceleratorBuffer;
    class Function;
}

namespace qcor {
namespace instructions {
class Exp: public xacc::IRGenerator {
public:
	std::shared_ptr<xacc::Function> generate(
			std::shared_ptr<xacc::AcceleratorBuffer> buffer,
			std::vector<xacc::InstructionParameter> parameters = std::vector<
					xacc::InstructionParameter> { }) override;

	std::shared_ptr<xacc::Function> generate(
			std::map<std::string, xacc::InstructionParameter>& parameters) override;

	std::shared_ptr<xacc::Function> generate(
			std::map<std::string, xacc::InstructionParameter>&& parameters) override;

    bool validateOptions() override;

	const std::string name() const override {
		return "exp_i_theta";
	}

	const std::string description() const override {
		return "";
	}
};
}
}
#endif