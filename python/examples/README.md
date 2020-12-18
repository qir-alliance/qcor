Employing the QCOR just-in-time (qjit) compilation features, we can wrap QCOR in Python bindings. Here we give examples of how this is used.

`bell.py` gives a straightforward implementation of the bell state on a quantum computer. A good first example for new users.

`exp_i_theta.py` highlights operator exponentiation in qcor.

`multiple_kernel.py` shows how one might solve a problem using multiple quantum kernels. Shows kernel composition functionality.
 
`openfermion_integration.py` shows interop capabilities of [openFermion](https://github.com/quantumlib/OpenFermion) with QCOR. Shows how an `openFermion` `FermionOperator` can be used as a `qcor` `Operator` for variational problems.   

`qaoa_circuit.py` shows an implementation of the Quantum Approximate Eigensolver circuit and args translation functor in Python.

`qsim_example.py`shows workflow of quantum simulation (qsim) routines in qcor.

`qsim_vqe.py` qsim example of VQE routine for Deuteron Hamiltonian.

`qsim_deuteron_qaoa.py` qsim example of QAOA routine for Deuteron Hamiltonian.

`qsim_qite_simple.py` qsim example of QITE (Quantum Imaginary Time Evolution) routine for a simple Hamiltonian.

`qsim_adapt_openfermion.py` qsim example of Adapt-VQE routine for a simple FermionOperator Hamiltonian.

`qsim_heisenberg_model.py` qsim example of time-dependent simulation for a general Heisenberg Hamiltonian.

`vqe_qcor_spec.py` example of VQE using `taskInitiate` for asynchronous execution of quantum-classical hybrid computations. 

`pyscf_qubit_tapering.py` example of using the QCOR `OperatorTransform`, specifically running [Qubit Tapering](https://arxiv.org/abs/1701.08213) followed by VQE using the QSim library.

`bit_flip_code_ftqc.py` example using the QCOR ftqc runtime to support fault-tolerant, fast-feedback instruction execution. 

`vqe_ftqc.py` example demonstrating VQE algorithm using the ftqc runtime. 

`unitary.py` example demonstrating circuit synthesis language extension.