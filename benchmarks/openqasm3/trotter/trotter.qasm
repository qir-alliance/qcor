OPENQASM 3;

const nb_steps = 100;
const nb_qubits = 50;
const step_size = 0.01;
const Jz = 1.0;
const h = 1.0;

qubit r[nb_qubits];

// -h*sigma_x layers
for i in [0:nb_steps] {
  rx(-h * step_size) r[i];
}

// -Jz*sigma_z*sigma_z layers
for i in [0:nb_steps - 1] {
  cx r[i], r[i+1];
  rz(-Jz * step_size) r[i + 1];
  cx r[i], r[i+1];
}