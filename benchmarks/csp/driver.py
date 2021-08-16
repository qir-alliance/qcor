# Run circuit optimization benchmarks
import glob
import os
import sys
import subprocess
import pathlib
import csv
import re
import numpy as np

qcorExe = str(pathlib.Path.home()) + "/.xacc/bin/qcor"

# CSV result
# Headers:
# We collect two sets of data:
# (1) Each pass (optimizer) runs *independently* using the same input circuit.
# (2) Passes run in series.
# Passes to run: Staq's rotation-folding, single-qubit-gate-merge, circuit-optimizer, voqc
# Note: VOQC can only be executed if the corresponding XACC plugin is installed.
# Please see https://github.com/tnguyen-ornl/SQIR for installation instructions.     
headers = ["Test Case",
           # Independent passes
           "Staq-Walltime", "Staq-Before", "Staq-After",
           "Gate Merge-Walltime", "Gate Merge-Before", "Gate Merge-After",
           "Circuit Optimizer-Walltime", "Circuit Optimizer-Before", "Circuit Optimizer-After",
           "VOQC-Walltime", "VOQC-Before", "VOQC-After",
           # Passes in sequence
           "Staq-Walltime", "Staq-Before", "Staq-After",
           "Gate Merge-Walltime", "Gate Merge-Before", "Gate Merge-After",
           "Circuit Optimizer-Walltime", "Circuit Optimizer-Before", "Circuit Optimizer-After",
           "VOQC-Walltime", "VOQC-Before", "VOQC-After",
           ]

# Extract data from log and append data to CSV
firstWrite = True
resultCsvFn = "result.csv"

if os.path.exists(resultCsvFn):
  os.remove(resultCsvFn)

# Analyze the full benchmark data for one test case:
def analyzeLog(testCaseName, log):
  global firstWrite
  print(log)
  rowData = [os.path.splitext(testCaseName)[0]]
  for line in log.splitlines():
    # Get elapsed time:
    if "Elapsed time:" in line:
      splitResult = re.search("Elapsed time:(.*)\[ms\]", line)
      rowData.append(float(splitResult.group(1)))
    # Get before:
    if "Number of Gates Before:" in line:
      splitResult = re.search("Number of Gates Before: (.*)", line)
      rowData.append(int(splitResult.group(1)))
    # Get after:
    if "Number of Gates After:" in line:
      splitResult = re.search("Number of Gates After: (.*)", line)
      rowData.append(int(splitResult.group(1)))
  #print("Data: ", rowData)
  if (len(rowData) != len(headers)):
    Exception("Failed to parse result for test case " + testCaseName)

  with open('result.csv', 'a', newline='') as csvfile:
    resultWriter = csv.writer(csvfile)
    if firstWrite is True:
      resultWriter.writerow(headers)
      firstWrite = False
    # Write data
    resultWriter.writerow(rowData)

# Run a full benchmark suite (the folder in /resources directory)
# Options: staq, qasm, chemistry
# Notes:
# staq suite source: https://github.com/tnguyen-ornl/qcor/tree/tnguyen/opt-data/benchmarks/resources/staq
# qasm suite source: https://github.com/tnguyen-ornl/qcor/tree/tnguyen/opt-data/benchmarks/resources/qasm
# chemistry suite source: https://github.com/tnguyen-ornl/qcor/tree/tnguyen/opt-data/benchmarks/resources/chemistry
def runBenchmarkSuite(suiteName):
  # Run test cases
  dirPath = os.path.dirname(os.path.realpath(__file__))
  listOfSrcFiles = glob.glob(dirPath + "/resources/" + suiteName +"/*.qasm")
  listOfTestCases = []
  # Note: Uncomment the followings if you have input QASM files 
  # which contain scientific notation numbers.
  # Test circuits have those formatting issues fixed, hence comment out this block.
  # Fixed Staq bug which cannot handle scientific form
  # for srcFile in listOfSrcFiles:
  #   with open(srcFile, 'r') as file:
  #     data = file.read()
  #     match_number = re.compile("-?\d*\.?\d+e[+-]?\d+")
  #     e_list = [x for x in re.findall(match_number, data)]
  #     e_list = list(dict.fromkeys(e_list))
  #     floatList = []
  #     for num in e_list:
  #       floatList.append(np.format_float_positional(float(num)))
  #     for i in range(len(floatList)):
  #       # Replace scientific notation
  #       data = data.replace(e_list[i], floatList[i])
  #   if len(floatList) > 0:
  #     with open(srcFile, 'w') as newFile:
  #       newFile.write(data)

  for srcFile in listOfSrcFiles:
    testCaseName = os.path.splitext(os.path.basename(srcFile))[0] + ".out"
    # Don't run super long circuits
    count = len(open(srcFile).readlines())
    if count > 100000:
      print("Skip Benchmark: ", testCaseName)
      continue

    listOfTestCases.append(testCaseName)
    print("Run Benchmark: ", testCaseName)
    resultLog = ""
    # Run single independent passes:
    # Sequence: "rotation-folding", "gate merge", "circuit optimizer", "voqc"
    passesToRun = ["rotation-folding",
                  "single-qubit-gate-merging", "circuit-optimizer", "voqc"]
    for passName in passesToRun:
      # Single pass data:
      compileProcess = subprocess.Popen([qcorExe, "-DTEST_SOURCE_FILE=\"" + srcFile + "\"", "-o", testCaseName, "-opt-pass", passName, "-print-opt-stats", "circuit_opt_benchmark.cpp"],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
      stdout, stderr = compileProcess.communicate()
      if len(stderr) > 1:
        # raise Exception("Failed to compile test case " + testCaseName)
        print("Failed to compile test case " + testCaseName)
        print(stderr.decode("utf-8"))
        continue
      testExe = dirPath + "/" + testCaseName
      exeProcess = subprocess.Popen(
          [testExe, passName], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = exeProcess.communicate()
      #print(stdout.decode("utf-8"))
      resultLog += stdout.decode("utf-8")
      os.remove(testExe)

    # Run passes in sequence (opt-level = 1)
    # Configurations:
    # Optimization level to run:
    OPT_LEVEL = 1
    compileProcess = subprocess.Popen([qcorExe, "-DTEST_SOURCE_FILE=\"" + srcFile + "\"", "-o", testCaseName, "-opt", str(OPT_LEVEL), "-opt-pass", "voqc", "-print-opt-stats", "circuit_opt_benchmark.cpp"],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
    stdout, stderr = compileProcess.communicate()
    if len(stderr) > 1:
      raise Exception("Failed to compile test case " + testCaseName)
    testExe = dirPath + "/" + testCaseName
    exeProcess = subprocess.Popen(
        [testExe], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = exeProcess.communicate()
    resultLog += stdout.decode("utf-8")
    os.remove(testExe)
    # Now analyze the full log
    analyzeLog(testCaseName, resultLog)

# Run the suite: e.g. qasm, chemistry, staq
runBenchmarkSuite("staq")
