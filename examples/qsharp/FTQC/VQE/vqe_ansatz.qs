namespace QCOR 
{
open QCOR.Intrinsic;
// Running VQE with an externally-provided optimization "stepper" interface:
// The stepper interface is "(current_params, energy) => (new_params)"
// Note: parameter initialization is done here.
// Returns the final energy.
operation DeuteronVqe(shots: Int, stepper : ((Double, Double[]) => Double[])) : Double {
    // Stopping conditions:
    let max_iters = 10;
    let f_tol = 0.01;
    let initial_params = [1.23];
        
    mutable opt_params = initial_params;
    mutable energy_val = 0.0;
    use qubits = Qubit[2]
    {
        for iter_id in 1..max_iters {
            mutable numParityOnes = 0;
            for test in 1..shots {
                X(qubits[0]);
                Ry(opt_params[0], qubits[1]);
                CNOT(qubits[1], qubits[0]);
                // Let's measure <X0X1>
                H(qubits[0]);
                H(qubits[1]);
                if M(qubits[0]) != M(qubits[1]) 
                {
                    set numParityOnes += 1;
                }
                Reset(qubits[0]);
                Reset(qubits[1]);
            }
            set energy_val =  IntAsDouble(shots - numParityOnes)/IntAsDouble(shots) - IntAsDouble(numParityOnes)/IntAsDouble(shots);
            // Stepping...
            set opt_params = stepper(energy_val, opt_params);
        } 
    }

    // Final energy:
    return energy_val;
}
}