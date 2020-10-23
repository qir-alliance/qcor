#!/bin/bash
set - o errexit
java -jar antlr-4.8-complete.jar -Dlanguage=Cpp -visitor -o generated/ -package pyxasm pyxasm.g4
