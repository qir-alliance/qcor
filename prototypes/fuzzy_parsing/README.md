This directory provides an example for fuzzy parsing 
with Clang. It provides an ExternalSemaSource that will 
handle undeclared identifiers upon parsing quantum-specific 
code inside of quantum kernel lambdas. Furthermore, this example 
shows how to get the kernel lambda as a source string, which we can 
pass off to XACC.

To build run 'make', to execute 

$ build/fuzzy_parsing test.c
