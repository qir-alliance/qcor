# Run circuit optimization benchmarks
import glob
import os
import sys
import subprocess
import pathlib
qcorExe = str(pathlib.Path.home()) + "/.xacc/bin/qcor"

# Configurations:
# Optimization level to run:
OPT_LEVEL = 1

dirPath = os.path.dirname(os.path.realpath(__file__))
listOfSrcFiles = glob.glob(dirPath + "/resources/*.qasm")
listOfTestCases = []
for srcFile in listOfSrcFiles:
  testCaseName = os.path.splitext(os.path.basename(srcFile))[0] + ".out"
  listOfTestCases.append(testCaseName)
  print("Compile test case: ", testCaseName)

  process = subprocess.Popen([qcorExe, "-DTEST_SOURCE_FILE=\"" + srcFile + "\"", "-o", testCaseName, "-opt", str(OPT_LEVEL), "-print-opt-stats", "circuit_opt_benchmark.cpp"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()

for testCase in listOfTestCases:
  print("Run Benchmark: ", testCase)
  testExe = dirPath + "/" + testCase
  process = subprocess.run([testExe], stdout=sys.stdout)
