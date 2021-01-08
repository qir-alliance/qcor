Here we show a few examples of QCOR programs. These exampls highlight many aspects of the API and show how typical QCOR programs are formed. To run these, simply navigate to the desired directory and execute the `in_file.cpp` with `qcor -o out_file -qpu [qpu-name] in_file.cpp`.

`adapt/` examples using the [adaptive VQE algorithm](https://arxiv.org/abs/1911.10205).

`adder/` implementations for addition of classical numbers on QPUs.

`adjoint_test/` showcases QCOR's `adjoint` functionality, which maps a quantum kernel to its Hermitian conjugate. 

`bell/` contains examples generating the bell state on quantum computer. Highlights multi-kernel, multi-qreg use cases. 

`deuteron/` contains examples solving the ground state of the Deuteron Hamiltonian using hybrid, variational methods. Contains example for operator exponentiation.

`error_mitigation/` examples employing error mitigation strategies.

`ftqc_qrt/` contains various fault tolerant quantum computation routines employing error correction methods. Replies on QCOR quantum runtime (qrt) library. Compile with `qcor -qpu aer[noise-model:<noise.json>] -qrt ftqc [ftqc_qrt_file].cpp `. 

`grover/` contains an implementation of [Grover's algorithm](https://en.wikipedia.org/wiki/Grover%27s_algorithm) and a `.qasm` file for the Grover 5 qubit kernel. 

`hadamard_test/` shows examples of kernel composition with 'Hadamard' and 'X' gate kernels.

`hybrid/` highlights quantum-classical computation tasks in QCOR, such as VQE, QAOA, and ADAPT.

`placement/` contains examples highlighting user controls for mapping logical qubits to physical qubits on an accelerator.

`qaoa/` examples for the [Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028). Includes an example for the `arg_translation` functor for mapping results from an optimizer to the arguments for the kernel function.

`qjit/` highlights "just in time" compilation capabilities of QCOR. 

`qpe/` showcases examples for [quantum phase estimation](https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm) algorithm. Files Requiring quantum runtime library (qrt), compile with: `qcor -o qpe -qpu qpp -shots 1024 -qrt qpe_example_qrt.cpp`

`quasimo/` a suite of examples employing the "Quantum Simulation Modeling" library with qcor.  

`simple/` highlights core features and functionality of QCOR programs running straightforward quantum computing schemes. Reccomended starting point for new users. 

`unitary/` examples mapping a unitary matrix to its quantum circuit representation. 