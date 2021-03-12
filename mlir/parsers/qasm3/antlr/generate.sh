#!/bin/bash
set - o errexit
java -jar antlr-4.9.2-complete.jar -Dlanguage=Cpp -visitor -o generated/ -package qasm3 qasm3.g4
