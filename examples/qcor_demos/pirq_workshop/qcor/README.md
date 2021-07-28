# QCOR - XACC IR Demo


QCOR IDE set-up: https://aide-qc.github.io/deploy/getting_started/


## Launch QCOR IDE: 

```
aide-qc --start
```

## Access:

```
http://CADES-IP:8080
```


## IDE Docker set-up

- Add `.ibm_config`, `.ionq_config`, and `.qlm_config`

- Install:

```
python3 -m pip install qiskit --user
python3 -m pip install qlmaas --user
```

- Notes:
 
`-print-csp-source`: print CSP output

`-print-final-submission`: print final IR submission (NISQ) or apply (FTQC)

`-print-opt-stats`: print runtime pass execution info

# Demo1: 

- No opt: print-final-submission to show final XACC IR tree for backend submission

```
qcor simple_circuit.cpp -print-final-submission
```

- Opt: level 1

```
qcor -opt 1 simple_circuit.cpp -print-final-submission
```

- With opt stats:

```
qcor -opt 1 simple_circuit.cpp -print-final-submission -print-opt-stats
```


- Trotter example:

```
qcor trotter_decompose.cpp -print-final-submission 
```

```
./a.out -dt 0.05 -steps 10
```

```
qcor -opt 1 trotter_decompose.cpp -print-final-submission 
```

```
qcor -opt 2 trotter_decompose.cpp -print-final-submission 
```


# Demo2: 

- Simulator
```
qcor -qpu qpp ghz.cpp 
```

- IBMQ
```
qcor -qpu ibm:ibm_lagos ghz.cpp
```

```
qcor -qpu aer:ibm_lagos ghz.cpp
```

- IonQ: 
```
qcor -qpu ionq ghz.cpp
```

```
qcor -qpu ionq:qpu ghz.cpp
```

- Atos QLM:
```
qcor -qpu atos-qlm ghz.cpp
```
(25-q max)


# Demo3: 

```
qcor -qrt ftqc iqpe_ftqc.cpp
```

Show FTQC apply:
```
qcor -qrt ftqc iqpe_ftqc.cpp -print-final-submission
```