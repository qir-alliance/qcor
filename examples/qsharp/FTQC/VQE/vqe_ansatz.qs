namespace QCOR 
{
open QCOR.Intrinsic;
// Running VQE with an externally-provided optimization "stepper" interface:
// The stepper interface is "(current_params, energy) => (new_params)"
// Note: parameter initialization is done here.
// Returns the final energy.
operation DeuteronVqe(shots: Int, stepper : ((Double, Double[]) => Double[])) : Double {
    // Initial parameters
    let initial_params = [1.234];   
    mutable opt_params = initial_params;
    mutable energy_val = 0.0;
    use qubits = Qubit[2]
    {
        // Use repeat-until-success pattern:
        // when the optimization loop converges,
        // the stepper will return an empty param array.
        repeat {
            let xxExp = DeuteronXX(qubits, shots, opt_params[0]);
            let yyExp = DeuteronYY(qubits, shots, opt_params[0]);
            let z0_z1_exps = DeuteronZ0_Z1(qubits, shots, opt_params[0]);
            let z0Exp = z0_z1_exps[0];
            let z1Exp = z0_z1_exps[1];
            // Deuteron energy:
            // H = 5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1
            set energy_val = 5.907 - 2.1433 * xxExp - 2.1433 * yyExp + 0.21829 * z0Exp - 6.125 * z1Exp;
            // Stepping...
            set opt_params = stepper(energy_val, opt_params);
        }
        until (Length(opt_params) == 0);
    }
    // Final energy:
    return energy_val;
}

// Base ansatz:
operation ansatz(qubits: Qubit[], theta: Double) : Unit {
    X(qubits[0]);
    Ry(theta, qubits[1]);
    CNOT(qubits[1], qubits[0]);
}

// This is for testing only:
// We should use a nicer implementation...
// XX term
operation DeuteronXX(qubits: Qubit[], shots: Int, theta: Double) : Double {
    mutable numParityOnes = 0;
    for shot in 1..shots {
        ansatz(qubits, theta);
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
    
    let exp_val =  IntAsDouble(shots - numParityOnes)/IntAsDouble(shots) - IntAsDouble(numParityOnes)/IntAsDouble(shots);
    return exp_val;
}

// YY term
operation DeuteronYY(qubits: Qubit[], shots: Int, theta: Double) : Double {
    mutable numParityOnes = 0;
    for shot in 1..shots {
        ansatz(qubits, theta);
        // Let's measure <Y0Y1>
        Rx(1.57079632679, qubits[0]);
        Rx(1.57079632679, qubits[1]);
        if M(qubits[0]) != M(qubits[1]) 
        {
            set numParityOnes += 1;
        }
        Reset(qubits[0]);
        Reset(qubits[1]);
    }
    
    let exp_val =  IntAsDouble(shots - numParityOnes)/IntAsDouble(shots) - IntAsDouble(numParityOnes)/IntAsDouble(shots);
    return exp_val;
}

// Z0 and Z1 terms
operation DeuteronZ0_Z1(qubits: Qubit[], shots: Int, theta: Double) : Double[] {
    mutable numParityOnesZ0 = 0;
    mutable numParityOnesZ1 = 0;

    for shot in 1..shots {
        ansatz(qubits, theta);
        if M(qubits[0]) == One
        {
            set numParityOnesZ0 += 1;
        }
        if M(qubits[1]) == One
        {
            set numParityOnesZ1 += 1;
        }
        Reset(qubits[0]);
        Reset(qubits[1]);
    }
    
    let exp_val_z0 =  IntAsDouble(shots - numParityOnesZ0)/IntAsDouble(shots) - IntAsDouble(numParityOnesZ0)/IntAsDouble(shots);
    let exp_val_z1 =  IntAsDouble(shots - numParityOnesZ1)/IntAsDouble(shots) - IntAsDouble(numParityOnesZ1)/IntAsDouble(shots);
    return [exp_val_z0, exp_val_z1];
}
}