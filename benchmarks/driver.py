# Run circuit optimization benchmarks
import glob
import os
import sys
import subprocess
import pathlib
import csv
import re

qcorExe = str(pathlib.Path.home()) + "/.xacc/bin/qcor"

# CSV result
# Headers:
headers = ["Test Case", 
          "Staq-Walltime", "Staq-Before", "Staq-After", 
          "Gate Merge-Walltime", "Gate Merge-Before", "Gate Merge-After", 
          "Circuit Optimizer-Walltime", "Circuit Optimizer-Before", "Circuit Optimizer-After", 
          "XACC Built-in Combined-Walltime", "XACC Built-in Combined-Before", "XACC Built-in Combined-After", 
          "Level 1 Combined-Walltime", "Level 1 Combined-Before", "Level 1 Combined-After"]

# Extract data from log and append data to CSV
firstWrite = True
resultCsvFn = "result.csv"

if os.path.exists(resultCsvFn):
  os.remove(resultCsvFn)

def analyzeLog(testCaseName, log):
  global firstWrite
  rowData = [os.path.splitext(testCaseName)[0]]
  for line in log.splitlines():
    # print("Receive '", line)
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
  print("Data: ", rowData)
  with open('result.csv', 'a', newline='') as csvfile:
    resultWriter = csv.writer(csvfile)
    if firstWrite is True:
      resultWriter.writerow(headers)
      firstWrite = False
    # Write data
    resultWriter.writerow(rowData)

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
  analyzeLog(testCaseName, stdout.decode("utf-8")) 
  os.remove(testExe)
  
