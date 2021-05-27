#include <fstream>
#include <iostream>
#include <sstream>

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "qcor_clang_wrapper.hpp"

using namespace clang::driver;
using namespace clang;
using namespace llvm;

namespace qcor {

llvm::ExitOnError ExitOnErr;
std::string GetExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

std::unique_ptr<clang::CodeGenAction> emit_llvm_ir(
    const std::string src_code, std::vector<std::string> extra_headers) {
  // Persist the src code to a temporary file
  std::string internal_file_name = ".__qcor_internal_llvm_ir_emitter.cpp";
  std::ofstream file(internal_file_name);
  file << src_code;
  file.close();

  // Define the Clang command line
  std::vector<std::string> argv_vec{"@CLANG_EXECUTABLE@", "-std=c++17"};
  std::vector<std::string> base_includes{
      "-I@XACC_ROOT@/include/xacc", "-I@XACC_ROOT@/include/pybind11/include",
      "-I@CMAKE_INSTALL_PREFIX@/include/qcor",
      "-I@XACC_ROOT@/include/quantum/gate", "-I@XACC_ROOT@/include/eigen",
      "-I@XACC_ROOT@/include/argparse"
      };
  for (auto extra : extra_headers) {
    base_includes.push_back(extra);
  }
  for (auto include : base_includes) {
    argv_vec.push_back(include);
  }
  argv_vec.push_back("-c");
  argv_vec.push_back(internal_file_name.c_str());

  std::vector<const char *> tmp_argv(argv_vec.size(), nullptr);
  for (int i = 0; i < argv_vec.size(); i++) {
    tmp_argv[i] = argv_vec[i].c_str();
  }

  // Create argc and argv
  const char **argv = &tmp_argv[0];
  int argc = argv_vec.size();
  void *MainAddr = (void *)(intptr_t)GetExecutablePath;
  std::string Path = "@LLVM_INSTALL_PREFIX@/bin/clang++";
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

  const std::string TripleStr = llvm::sys::getProcessTriple();
  llvm::Triple T(TripleStr);

  ExitOnErr.setBanner("clang interpreter");

  // Create the Clang driver
  Driver TheDriver(Path, T.str(), Diags);
  TheDriver.setTitle("clang interpreter");
  TheDriver.setCheckInputsExist(false);

  // FIXME: This is a hack to try to force the driver to do something we can
  // recognize. We need to extend the driver library to support this use model
  // (basically, exactly one input, and the operation mode is hard wired).
  SmallVector<const char *, 16> Args(argv, argv + argc);
  // Args.push_back("-fsyntax-only");
  std::unique_ptr<Compilation> C(TheDriver.BuildCompilation(Args));
  if (!C) {
    std::cout << "QCOR internal clang execution error. Could not create the "
                 "Compilation data structure.\n";
    exit(1);
  }

  // We expect to get back exactly one command job, if we didn't something
  // failed. Extract that job from the compilation.
  const driver::JobList &Jobs = C->getJobs();
  if (Jobs.size() != 1 || !isa<driver::Command>(*Jobs.begin())) {
    SmallString<256> Msg;
    llvm::raw_svector_ostream OS(Msg);
    Jobs.Print(OS, "; ", true);
    Diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
    std::cout << "Error in creating the clang JobList.\n";
    exit(1);
  }

  const driver::Command &Cmd = cast<driver::Command>(*Jobs.begin());
  if (llvm::StringRef(Cmd.getCreator().getName()) != "clang") {
    Diags.Report(diag::err_fe_expected_clang_command);
    std::cout << "Error in creating the clang driver command.\n";
  }

  // Initialize a compiler invocation object from the clang (-cc1) arguments.
  const llvm::opt::ArgStringList &CCArgs = Cmd.getArguments();
  std::unique_ptr<CompilerInvocation> CI(new CompilerInvocation);
  CompilerInvocation::CreateFromArgs(*CI, CCArgs, Diags);

  // Create a compiler instance to handle the actual work.
  CompilerInstance Clang;
  Clang.setInvocation(std::move(CI));

  // Create the compilers actual diagnostics engine.
  Clang.createDiagnostics();
  if (!Clang.hasDiagnostics()) {
    std::cout << "Error - could not create Clang diagnostics.\n";
    exit(1);
  }

  // Infer the builtin include path if unspecified.
  if (Clang.getHeaderSearchOpts().UseBuiltinIncludes &&
      Clang.getHeaderSearchOpts().ResourceDir.empty())
    Clang.getHeaderSearchOpts().ResourceDir =
        CompilerInvocation::GetResourcesPath(argv[0], MainAddr);

  // Create and execute the frontend to generate an LLVM bitcode module.
  std::unique_ptr<CodeGenAction> Act(new EmitLLVMOnlyAction());
  if (!Clang.ExecuteAction(*Act)) {
    std::cout << "Error in executing clang codegen.\n";
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  std::remove(internal_file_name.c_str());

  return Act;
}

}  // namespace qcor
