OPENQASM 3;
include "stdgates.inc";

const n_qubits = 50;
const n_layers = 4;
qubit q[n_qubits];

// Run by: mpiexec -n <N> ./a.out -qrt nisq -qpu tnqvm -qpu-config tnqvm.ini 

// Loop over layers
float[64] theta = 1.234;
for i in [0:n_layers] {
    // Single qubit layers:
    for j in [0:n_qubits] {
        rx(theta) q[j];
    }

    // For demonstration purposes, just change the 
    // angle in each layer by adding 1.0.~
    theta += 1.0;
    // Entanglement layers:
    for j in [0:n_qubits - 1] {
        cx q[j], q[j+1];
    }
}

// 4 Ranks, 12 layers
// real    8m44.685s
// user    68m45.740s
// sys     0m14.613s
// Result Buffer:
// {
//     "AcceleratorBuffer": {
//         "name": "",
//         "size": 50,
//         "Information": {
//             "amplitude-imag": 6.013328501808246e-9,
//             "amplitude-real": -9.093254149661334e-9
//         },
//         "Measurements": {}
//     }
// }
//  Number of Flops processed    :      0.12953981373440D+13
//  Number of Flops processed    :      0.12953981373440D+13
//  Average GEMM GFlop/s rate    :      0.34592207040862D+01
//  Average GEMM GFlop/s rate    :      0.33989965113118D+01
//  Number of Bytes permuted     :      0.13790950400000D+11
//  Average permute GB/s rate    :      0.46258781869063D-01
//  Number of Bytes permuted     :      0.13790950400000D+11
//  Average contract GFlop/s rate:      0.15991027746986D+01
//  Average permute GB/s rate    :      0.45411628655322D-01
// #END_MSG
//  Average contract GFlop/s rate:      0.15672632853478D+01
// #END_MSG
// #MSG(TAL-SH::CP-TAL): Statistics on CPU:
//  Number of Flops processed    :      0.12953981373440D+13
//  Average GEMM GFlop/s rate    :      0.34815246662248D+01
//  Number of Bytes permuted     :      0.13790950400000D+11
//  Average permute GB/s rate    :      0.45497061206249D-01
//  Average contract GFlop/s rate:      0.15932956878495D+01
// #END_MSG
// #MSG(TAL-SH::CP-TAL): Statistics on CPU:
//  Number of Flops processed    :      0.12953981373440D+13
//  Average GEMM GFlop/s rate    :      0.34580808710314D+01
//  Number of Bytes permuted     :      0.13790950400000D+11
//  Average permute GB/s rate    :      0.46148523562615D-01
//  Average contract GFlop/s rate:      0.15968730806935D+01
// #END_MSG