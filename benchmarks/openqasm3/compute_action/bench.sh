#!/bin/bash

# From 6 to 50 qubits, even only, 
# create the benchmark qasm code, compile it with -O3
# and execute collecting total number of gates
# Change 'Total G' to 'Total C' to count total 
# ctrl opertions
for NQUBITS in `seq 6 2 50`
do
	sed "s/nb_qubits =.*/nb_qubits = $NQUBITS;/" bench.qasm > bench$NQUBITS.qasm 
    qcor -O3 bench$NQUBITS.qasm -qrt ftqc -qpu tracer
    ./a.out | grep 'Total G' | sed 's/.*\: //'
done
