Employing the QCOR just-in-time (qjit) compilation features, we can wrap QCOR in Python bindings. Here we give examples of how this is used.

`bell.py` gives a straightforward implementation of the bell state on a quantum computer. A good first example for new users.

`exp_i_theta.py` highlights operator exponentiation in qcor.

`multiple_kernel.py` shows how one might solve a problem using multiple quantum kernels. Shows kernel composition functionality.
 
`openfermion_integration.py` shows interop capabilities of [openFermion](https://github.com/quantumlib/OpenFermion) with QCOR. Shows how an `openFermion` `FermionOperator` can be used as a `qcor` `Operator` for variational problems.   

`qaoa_circuit.py` shows an implementation of the Quantum Approximate Eigensolver circuit and args translation functor in Python.

`qsim_example.py`shows workflow of quantum simulation (qsim) routines in qcor.

`qsim_vqe.py` qsim example of VQE routine for Deuteron Hamiltonian.

`vqe_qcor_spec.py` example of VQE using `taskInitiate` for asynchronous execution of quantum-classical hybrid computations. 