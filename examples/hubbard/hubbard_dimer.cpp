// hubbard_dimer_ex.cpp

using FO = FermionOperator;
// define factorized UCC ansatz using exp_i_theta
__qpu__ void ansatz(qreg q, std::vector<double> x, std::vector<FO> ops, int order, int layer) {
  // prepare reference state |1010> (qubit map: 0↑,1↑,0↓,1↓ -> 0,1,2,3)
  X(q[0]);
  X(q[2]);
  // define a few different orders of UCC factors 
  // "1": singles, "2": doubles
  // default 211: doubles term first, then two singles terms
  std::vector<int> idx;
  if (order == 211) {
    idx = {0, 1, 2};
  } else if (order == 112) {
    idx = {1, 2, 0};
  } else if (order == 121) {
    idx = {1, 0, 2};
  }
  // default 1 layer per UCC factor; 2 layers necessary for 112 order
  for (int n = 0; n < layer; n++) {
    for (int i = 0; i < ops.size(); i++) {
      exp_i_theta(q, x[idx[i]], ops[idx[i]]);
    }
  }
}

int main(int argc, char **argv) {
  // handle command line arguments
  // --t: hopping integral
  // --U: Coulomb interaction
  // --order: sequence order of 3 factors of UCC ansatz
  // --layer: number of layers of 3-factor UCC ansatz  
  double t = 1;
  double U = 3;
  int ucc_order = 211;  // "1": singles, "2": doubles
  int ucc_layer = 1;
  int n_params = 3;
  std::vector<double> init_params = {0., 0., 0.};
  std::vector<std::string> arguments(argv + 1, argv + argc);
  for (int i = 0; i < (int)arguments.size()-1; i++) {
    if (arguments[i] == "--U") {
      U = std::stod(arguments[i + 1]);
      std::cout << "recieved U = " << U << "\n";
    } else if (arguments[i] == "--t") {
      t = std::stod(arguments[i + 1]);
      std::cout << "recieved t = " << t << "\n";    
    } else if (arguments[i] == "--order") {
      ucc_order = std::stoi(arguments[i + 1]);
      std::cout << "recieved ucc_order = " << ucc_order << "\n";
    } else if (arguments[i] == "--layer") {
      ucc_layer = std::stoi(arguments[i + 1]);
      std::cout << "recieved ucc_layer = " << ucc_layer << "\n";
    }
  }
  double mu = U / 2;
  int Max_layer = 20;
  try {
    int err[2] = {0, 0};
    if (!(ucc_order == 112 || ucc_order == 121 || ucc_order == 211)) {
      err[0] = 1;
    }
    if (ucc_layer < 1 || ucc_layer > Max_layer) {
      err[1] = 1;
    }
    if (err[0] || err[1]) {
      throw err;
    }
  } catch (int err[]) {
    if (err[0]) {
      printf("Exception: sequence order of 3 factors of UCC ansatz should be "
             "112, 121, or 211.\n");
    }
    if (err[1]) {
      printf("Exception: number of 3-factor UCC ansatz layers should be 1, "
             "..., %d.\n",
             Max_layer);
    }
    return 0;
  }

  // define Hamiltonian and UCC ops with QCOR API:
  // H defined using fermion creation operator adag() and annihilation operator a()
  //*
  auto H = -t * (adag(0) * a(1) + adag(1) * a(0) + adag(2) * a(3) + adag(3) * a(2)) +
      U * (adag(0) * a(0) * adag(2) * a(2) + adag(1) * a(1) * adag(3) * a(3)) -
      mu * (adag(0) * a(0) + adag(1) * a(1) + adag(2) * a(2) + adag(3) * a(3));
  //*/
  // H defined using FermionOperator()
  /*
  auto H = -t * (FO("0^ 1") + FO("1^ 0") + FO("2^ 3") + FO("3^ 2")) +
           U * (FO("0^ 0 2^ 2") + FO("1^ 1 3^ 3")) -
           mu * (FO("0^ 0") + FO("1^ 1") + FO("2^ 2") + FO("3^ 3"));
  */
  // H defined using Pauli operators X(), Y(), Z()
  /*
  auto H = -t * 0.5 * (X(0) * X(1) + Y(0) * Y(1) + X(2) * X(3) + Y(2) * Y(3)) +
      U * 0.25 * (2. + Z(0) * Z(2) + Z(1) * Z(3) - Z(0) - Z(1) - Z(2) - Z(3)) -
      mu * 0.5 * (4. - Z(0) - Z(1) - Z(2) - Z(3));
  */
  // define 3 factors of UCC ansatz:
  // ops[0]: doubles -- pair hopping (0↑,0↓) <-> (1↑,1↓)
  // ops[1]: singles -- hopping 0↑ <-> 1↑
  // ops[2]: singles -- hopping 0↓ <-> 1↓
  std::vector<FO> ops{FO("1^ 3^ 2 0") - FO("0^ 2^ 3 1"),
                      FO("1^ 0") - FO("0^ 1"), FO("3^ 2") - FO("2^ 3")};

  // translate arguments for use with ObjectiveFunction
  // map signature of ansatz to signature of ObjectiveFunction
  auto quantum_reg = qalloc(4);
  auto args_translation = std::make_shared<
      ArgsTranslator<qreg, std::vector<double>, std::vector<FO>, int, int>>(
      [&](const std::vector<double> x) {
        return std::tuple(quantum_reg, x, ops, ucc_order, ucc_layer);
      });
  // define the optimizer and optimization algorithm to use
  auto optimizer = createOptimizer(
      "nlopt", {{"algorithm", "l-bfgs"}, {"initial-parameters", init_params}});
  // define variational objective function
  auto objective = createObjectiveFunction(
      ansatz, H, args_translation, quantum_reg, n_params,
      {
          //{"gradient-strategy", "parameter-shift"}
          {"gradient-strategy", "central"},{"step", 0.001}
          //{"gradient-strategy", "autodiff"}
      });
  // start hybrid computation task
  auto [optval, opt_params] = optimizer->optimize(*objective.get());
  // print results
  printf("U = %g, t = %g, UCC factor order = %d, number of layers = %d\n", U, t,
         ucc_order, ucc_layer);
  printf("min<H>([%.10f,%.10f,%.10f]) = %.10f\n", opt_params[0], opt_params[1],
         opt_params[2], optval);
  return 0;
}
