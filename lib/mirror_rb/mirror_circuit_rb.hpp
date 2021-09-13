#pragma once
#include <memory>
#include <vector>
#include <string>
#include <utility>

// !Temporary: we'll figure out an interface for this later:
namespace qcor {
class CompositeInstruction;
// Return the 'mirrored' circuit along with the expected result.
std::pair<std::shared_ptr<CompositeInstruction>, std::vector<bool>>
createMirrorCircuit(std::shared_ptr<CompositeInstruction> in_circuit);
} // namespace qcor
