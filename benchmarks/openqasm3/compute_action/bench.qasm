OPENQASM 3;

const nb_qubits = 5;

def oracle() qubit[nb_qubits]:r {
  int nb_steps = 100;
  double step_size = .01;
  double Jz = 1.0;
  double h = 1.0;
  // -h*sigma_x layers
  for step in [0:nb_steps] {
    // -h*sigma_x layers
    rx(-h * step_size) r;
    
    // -Jz*sigma_z*sigma_z layers
    for i in [0:nb_qubits - 1] {
      cx r[i], r[i+1];
      rz(-Jz * step_size) r[i + 1];
      cx r[i], r[i+1];
    } 
  }
}

def oracle_compute_action() qubit[nb_qubits]:r {
  int nb_steps = 100;
  double step_size = .01;
  double Jz = 1.0;
  double h = 1.0;

  // -h*sigma_x layers
  for step in [0:nb_steps] {
    // -h*sigma_x layers
    rx (-h * step_size) r;

    // -Jz*sigma_z*sigma_z layers
    for i in [0:nb_qubits - 1] {
      compute {
        cx r[i], r[i+1];
      } action {
        rz(-Jz * step_size) r[i + 1];
      }
    } 
  }
}

qubit r[nb_qubits], c;

ctrl @ oracle_compute_action c, r;
// ctrl @ oracle c, r;
