# Run circuit optimization benchmarks
import glob
import os
import sys
import subprocess
import pathlib
qcorExe = str(pathlib.Path.home()) + "/.xacc/bin/qcor"

# CSV result
# Headers:
resultCsv = "Test Case, Staq, Circuit Optimizer, Gate Merge, Level 1 Combined, \n"

# Extract data from log and append data to CSV
def analyzeLog(log):
  # TODO
  pass

# Configurations:
# Optimization level to run:
# TODO: enable single-pass execution
OPT_LEVEL = 1

dirPath = os.path.dirname(os.path.realpath(__file__))
listOfSrcFiles = glob.glob(dirPath + "/resources/qasm/*.qasm")
listOfTestCases = []
for srcFile in listOfSrcFiles:
  testCaseName = os.path.splitext(os.path.basename(srcFile))[0] + ".out"
  listOfTestCases.append(testCaseName)
  print("Compile test case: ", testCaseName)
  compileProcess = subprocess.Popen([qcorExe, "-DTEST_SOURCE_FILE=\"" + srcFile + "\"", "-o", testCaseName, "-opt", str(OPT_LEVEL), "-print-opt-stats", "circuit_opt_benchmark.cpp"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  stdout, stderr = compileProcess.communicate()
  if len(stderr) > 1:
    raise Exception("Failed to compile test case " + testCaseName)
  print("Run Benchmark: ", testCaseName)
  testExe = dirPath + "/" + testCaseName
  exeProcess = subprocess.Popen([testExe], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = exeProcess.communicate()
  analyzeLog(stdout.decode("utf-8")) 
  os.remove(testExe)
  