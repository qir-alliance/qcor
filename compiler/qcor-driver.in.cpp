#include "qcor_frontend_action.hpp"

int main(int argc, char **argv) {

  xacc::Initialize();

  // Get filename
  // FIXME, we assume it is the last arg...
  std::string fileName(argv[argc - 1]);
  if (!xacc::fileExists(fileName)) {
    xacc::error("File " + fileName + " does not exist.");
  }

  std::ifstream t(fileName);
  std::string src((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());

  // Initialize rewriter
  Rewriter Rewrite;

  std::vector<std::string> args{"-Wno-dangling", "-std=c++14", 
                                "-I@CMAKE_INSTALL_PREFIX@/include/qcor",
                                "-I@CMAKE_INSTALL_PREFIX@/include/xacc"};

  // Do a little bit of argument analysis
  // We need to know any new include paths
  // and the accelerator backend
  std::string accName = "";
  std::vector<std::string> arguments(argv + 1, argv + argc);
  for (int i = 0; i < arguments.size(); i++) {
    if (arguments[i] == "-I") {
      if (arguments[i + 1] != "@CMAKE_INSTALL_PREFIX@/include/qcor" &&
          arguments[i + 1] != "@CMAKE_INSTALL_PREFIX@/include/xacc") {
        args.push_back(arguments[i] + arguments[i + 1]);
      }
    } else if (arguments[i].find("-I") != std::string::npos) {
      args.push_back(arguments[i]);
    } else if (arguments[i] == "--accelerator") {
      accName = arguments[i + 1];
    } else if (arguments[i] == "-a") {
      accName = arguments[i + 1];
    }
  }

  auto action = new qcor::compiler::QCORFrontendAction(Rewrite, fileName, args);

  if (!accName.empty()) {
    xacc::setAccelerator(accName);
  }
  if (!tooling::runToolOnCodeWithArgs(action, src, args)) {
    xacc::error("Error running qcor compiler.");
  }

  return 0;
}
