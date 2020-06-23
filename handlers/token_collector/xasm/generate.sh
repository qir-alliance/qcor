#!/bin/bash
set - o errexit
java -jar antlr-4.7.2-complete.jar -Dlanguage=Cpp -visitor -o generated/ -package xasm xasm_single.g4
