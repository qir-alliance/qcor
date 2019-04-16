#include "LibIntWrapper.hpp"
#include "libint2.hpp"
#include "libint2/basis.h"
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/KroneckerProduct>

#include <fstream>

namespace libintwrapper {
Eigen::MatrixXd compute_2body_fock(const libint2::BasisSet &shells,
                                   const Eigen::MatrixXd &D) {
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  auto time_elapsed = std::chrono::duration<double>::zero();

  const auto n = shells.nbf();
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);

  // construct the 2-electron repulsion integrals engine
  Engine engine(Operator::coulomb, shells.max_nprim(), shells.max_l(), 0);

  auto shell2bf = shells.shell2bf();

  const auto &buf = engine.results();

  // loop over permutationally-unique set of shells
  for (auto s1 = 0; s1 != shells.size(); ++s1) {

    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto n1 = shells[s1].size();   // number of basis functions in this shell

    for (auto s2 = 0; s2 <= s1; ++s2) {

      auto bf2_first = shell2bf[s2];
      auto n2 = shells[s2].size();

      for (auto s3 = 0; s3 <= s1; ++s3) {

        auto bf3_first = shell2bf[s3];
        auto n3 = shells[s3].size();

        const auto s4_max = (s1 == s3) ? s2 : s3;
        for (auto s4 = 0; s4 <= s4_max; ++s4) {

          auto bf4_first = shell2bf[s4];
          auto n4 = shells[s4].size();

          // compute the permutational degeneracy (i.e. # of equivalents) of the
          // given shell set
          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
          auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

          const auto tstart = std::chrono::high_resolution_clock::now();

          engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
          const auto *buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          const auto tstop = std::chrono::high_resolution_clock::now();
          time_elapsed += tstop - tstart;

          for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for (auto f2 = 0; f2 != n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;

                  const auto value = buf_1234[f1234];

                  const auto value_scal_by_deg = value * s1234_deg;

                  G(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
                  G(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
                  G(bf1, bf3) -= 0.25 * D(bf2, bf4) * value_scal_by_deg;
                  G(bf2, bf4) -= 0.25 * D(bf1, bf3) * value_scal_by_deg;
                  G(bf1, bf4) -= 0.25 * D(bf2, bf3) * value_scal_by_deg;
                  G(bf2, bf3) -= 0.25 * D(bf1, bf4) * value_scal_by_deg;
                }
              }
            }
          }
        }
      }
    }
  }

  // symmetrize the result and return
  Eigen::MatrixXd Gt = G.transpose();
  return 0.5 * (G + Gt);
}
void LibIntWrapper::generate(std::map<std::string, std::string> &&options) {
  generate(options);
}

void LibIntWrapper::generate(std::map<std::string, std::string> &options) {

  auto basis = options["basis"];
  auto geom = options["geometry"];

  std::ofstream tmp(".chem_obs_geom.xyz");
  tmp << geom;
  tmp.close();
  std::ifstream input_file(".chem_obs_geom.xyz");

  auto atoms = libint2::read_dotxyz(input_file);

  std::remove(".chem_obs_geom.xyz");

  // count the number of electrons
  auto nelectron = 0;
  for (auto i = 0; i < atoms.size(); ++i)
    nelectron += atoms[i].atomic_number;
  const auto ndocc = nelectron / 2;
//   std::cout << "# of electrons = " << nelectron << std::endl;

  auto enuc = 0.0;
  for (auto i = 0; i < atoms.size(); i++) {
    for (auto j = i + 1; j < atoms.size(); j++) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = std::sqrt(r2);
      enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
    }
  }
//   std::cout << "Nuclear repulsion energy = " << std::setprecision(15) << enuc
//             << std::endl;

  libint2::Shell::do_enforce_unit_normalization(false);

//   std::cout << "Atomic Cartesian coordinates (a.u.):" << std::endl;
//   for (const auto &a : atoms) {
//     std::cout << a.atomic_number << " " << a.x << " " << a.y << " " << a.z
//               << std::endl;
//   }

  libint2::initialize();

  libint2::BasisSet obs(basis, atoms, true);
  std::copy(begin(obs), end(obs),
            std::ostream_iterator<libint2::Shell>(std::cout, "\n"));

  size_t nao = 0;
  for (auto s = 0; s < obs.size(); ++s)
    nao += obs[s].size();

  libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(),
                             obs.max_l(), 0);

  libint2::Engine s_engine(libint2::Operator::overlap, obs.max_nprim(),
                           obs.max_l());
  libint2::Engine kinetic_engine(libint2::Operator::kinetic, obs.max_nprim(),
                                 obs.max_l());
  libint2::Engine nuclear_engine(libint2::Operator::nuclear, obs.max_nprim(),
                                 obs.max_l());
  nuclear_engine.set_params(make_point_charges(atoms));

  auto shell2bf = obs.shell2bf();
  const auto &buf = eri_engine.results();

  // const auto& is very important!
  nBasis = obs.nbf();
  auto nshells = obs.size();
//   std::cout << "NBF: " << nBasis << "\n";

  Eigen::Tensor<double, 1> data_(std::pow(nBasis, 4));
  int ii = 0;
  //   std::cout << "MATRIX:\n" << result << "\n";
  for (auto s1 = 0; s1 != obs.size(); ++s1) {
    for (auto s2 = 0; s2 != obs.size(); ++s2) {
      for (auto s3 = 0; s3 != obs.size(); ++s3) {
        for (auto s4 = 0; s4 != obs.size(); ++s4) {

        //   std::cout << "compute shell set {" << s1 << "," << s2 << "," << s3
                    // << "," << s4 << "} ... ";
          eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
        //   std::cout << "done. ";

          auto ints_shellset = buf[0];

        //   std::cout << ii << ", " << ints_shellset[0] << "\n";
          data_(ii) = ints_shellset[0];
          eri_data.push_back(ints_shellset[0]);
          ii++;
        }
      }
    }
  }

//   std::cout << "ERI Vec:\n" << data_ << "\n";

//   std::cout << "obs size: " << obs.size() << "\n";
  const auto &kinetic_buf_vec =
      kinetic_engine.results(); // will point to computed shell sets
                                // const auto& is very important!

  Eigen::Tensor<double, 2> T(nBasis, nBasis);
  for (auto s1 = 0; s1 != obs.size(); ++s1) {
    for (auto s2 = 0; s2 != obs.size(); ++s2) {

    //   std::cout << "T compute shell set {" << s1 << "," << s2 << "} ... ";
      kinetic_engine.compute(obs[s1], obs[s2]);
    //   std::cout << "done" << std::endl;

      auto ints_shellset =
          kinetic_buf_vec[0]; // location of the computed integrals
      if (ints_shellset == nullptr)
        continue; // nullptr returned if the entire shell-set was screened out

      auto bf1 = shell2bf[s1];  // first basis function in first shell
      auto n1 = obs[s1].size(); // number of basis functions in first shell
      auto bf2 = shell2bf[s2];  // first basis function in second shell
      auto n2 = obs[s2].size(); // number of basis functions in second shell

      // integrals are packed into ints_shellset in row-major (C) form
      // this iterates over integrals in this order
      for (auto f1 = 0; f1 != n1; ++f1)
        for (auto f2 = 0; f2 != n2; ++f2) {
          T(bf1 + f1, bf2 + f2) = ints_shellset[f1 * n2 + f2];
          kinetic_data.push_back(ints_shellset[f1 * n2 + f2]);

        //   std::cout << "  " << bf1 + f1 << " " << bf2 + f2 << " "
        //             << ints_shellset[f1 * n2 + f2] << std::endl;
        }
    }
  }

  //   std::cout << "T:\n" << T << "\n";

  const auto &nuclear_buf_vec =
      nuclear_engine.results(); // will point to computed shell sets
                                // const auto& is very important!

  Eigen::Tensor<double, 2> V(nBasis, nBasis);
  for (auto s1 = 0; s1 != obs.size(); ++s1) {
    for (auto s2 = 0; s2 != obs.size(); ++s2) {

    //   std::cout << "V compute shell set {" << s1 << "," << s2 << "} ... ";
      nuclear_engine.compute(obs[s1], obs[s2]);
    //   std::cout << "done" << std::endl;

      auto ints_shellset =
          nuclear_buf_vec[0]; // location of the computed integrals
      if (ints_shellset == nullptr)
        continue; // nullptr returned if the entire shell-set was screened out

      auto bf1 = shell2bf[s1];  // first basis function in first shell
      auto n1 = obs[s1].size(); // number of basis functions in first shell
      auto bf2 = shell2bf[s2];  // first basis function in second shell
      auto n2 = obs[s2].size(); // number of basis functions in second shell

      // integrals are packed into ints_shellset in row-major (C) form
      // this iterates over integrals in this order
      for (auto f1 = 0; f1 != n1; ++f1)
        for (auto f2 = 0; f2 != n2; ++f2) {
          V(bf1 + f1, bf2 + f2) = ints_shellset[f1 * n2 + f2];
          potential_data.push_back(ints_shellset[f1 * n2 + f2]);
        //   std::cout << "  " << bf1 + f1 << " " << bf2 + f2 << " "
        //             << ints_shellset[f1 * n2 + f2] << std::endl;
        }
    }
  }

//   std::cout << "V:\n" << V << "\n";

  const auto &overlap_buf_vec =
      s_engine.results(); // will point to computed shell sets
                          // const auto& is very important!

  Eigen::MatrixXd S(nBasis, nBasis);
  S.setZero();
  for (auto s1 = 0; s1 != obs.size(); ++s1) {
    for (auto s2 = 0; s2 != obs.size(); ++s2) {

    //   std::cout << "S compute shell set {" << s1 << "," << s2 << "} ... ";
      s_engine.compute(obs[s1], obs[s2]);
    //   std::cout << "done" << std::endl;

      auto ints_shellset =
          overlap_buf_vec[0]; // location of the computed integrals
      if (ints_shellset == nullptr)
        continue; // nullptr returned if the entire shell-set was screened out

      auto bf1 = shell2bf[s1];  // first basis function in first shell
      auto n1 = obs[s1].size(); // number of basis functions in first shell
      auto bf2 = shell2bf[s2];  // first basis function in second shell
      auto n2 = obs[s2].size(); // number of basis functions in second shell

      // integrals are packed into ints_shellset in row-major (C) form
      // this iterates over integrals in this order
      for (auto f1 = 0; f1 != n1; ++f1)
        for (auto f2 = 0; f2 != n2; ++f2) {
          S(bf1 + f1, bf2 + f2) = ints_shellset[f1 * n2 + f2];
          //   kinetic_data.push_back(ints_shellset[f1 * n2 + f2]);

        //   std::cout << "  " << bf1 + f1 << " " << bf2 + f2 << " "
        //             << ints_shellset[f1 * n2 + f2] << std::endl;
        }
    }
  }

//   std::cout << "S:\n" << S << "\n";

  Eigen::MatrixXd T_ =
      Eigen::Map<Eigen::MatrixXd>(T.data(), T.dimension(0), T.dimension(1));
  Eigen::MatrixXd V_ =
      Eigen::Map<Eigen::MatrixXd>(V.data(), V.dimension(0), V.dimension(1));

  Eigen::MatrixXd H = T_ + V_;

  Eigen::MatrixXd D;
  // solve H C = e S C
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> gen_eig_solver(H,
                                                                           S);
  auto eps = gen_eig_solver.eigenvalues();
  auto C = gen_eig_solver.eigenvectors();
//   std::cout << "\n\tInitial C Matrix:\n";
//   std::cout << C << std::endl;

  // compute density, D = C(occ) . C(occ)T
  auto C_occ = C.leftCols(ndocc);
//   std::cout << "COCC: \n" << C_occ << "\n";
  D = C_occ * C_occ.transpose();

//   std::cout << "\n\tInitial Density Matrix:\n";
//   std::cout << D << std::endl;

  /*** =========================== ***/
  /*** main iterative loop         ***/
  /*** =========================== ***/

  const auto maxiter = 100;
  const double conv = 1e-12;
  auto iter = 0;
  double rmsd = 0.0;
  double ediff = 0.0;
  double ehf = 0.0;
  do {
    ++iter;

    // Save a copy of the energy and the density
    auto ehf_last = ehf;
    auto D_last = D;

    // build a new Fock matrix
    auto F = H;
    // F += compute_2body_fock_simple(shells, D);
    F += compute_2body_fock(obs, D);

    // if (iter == 1) {
    //   std::cout << "\n\tFock Matrix:\n";
    //   std::cout << F << std::endl;
    // }

    // solve F C = e S C
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> gen_eig_solver(F,
                                                                             S);
    auto eps = gen_eig_solver.eigenvalues();
    C = gen_eig_solver.eigenvectors();

    // compute density, D = C(occ) . C(occ)T
    auto C_occ = C.leftCols(ndocc);
    D = C_occ * C_occ.transpose();

    // compute HF energy
    ehf = 0.0;
    for (auto i = 0; i < nao; i++)
      for (auto j = 0; j < nao; j++)
        ehf += D(i, j) * (H(i, j) + F(i, j));

    // compute difference with last iteration
    ediff = ehf - ehf_last;
    rmsd = (D - D_last).norm();

    if (iter == 1)
      std::cout << "\n\n Iter        E(elec)              E(tot)               "
                   "Delta(E)             RMS(D)         Time(s)\n";
    printf(" %02d %20.12f %20.12f %20.12f %20.12f\n", iter, ehf, ehf + enuc,
           ediff, rmsd);

  } while (((fabs(ediff) > conv) || (fabs(rmsd) > conv)) && (iter < maxiter));

  printf("** Hartree-Fock energy = %20.12f\n", ehf + enuc);

//   std::cout << "C:\n" << C << "\n\n\n";
  libint2::finalize(); // done with libint

  Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(2, 2);

  //    Eigen::Tensor<double,4> eye4 = eye.reshape(std::array<int,4>{1,1,2,2});
  Eigen::Tensor<double, 4> eri = getERI();

  Eigen::Tensor<double, 4> tmpK(nBasis, nBasis, 2 * nBasis, 2 * nBasis);
  tmpK.setZero();
  for (int i = 0; i < nBasis; i++) {
    for (int j = 0; j < nBasis; j++) {
      Eigen::MatrixXd tmp1(nBasis, nBasis);
      tmp1.setZero();
      for (int k = 0; k < nBasis; k++) {
        for (int l = 0; l < nBasis; l++) {
          tmp1(k, l) = eri(i, j, k, l);
        }
      }
      Eigen::MatrixXd kron = Eigen::kroneckerProduct(eye, tmp1).eval();

      for (int m = 0; m < 2 * nBasis; m++) {
        for (int n = 0; n < 2 * nBasis; n++) {
          tmpK(i, j, m, n) = kron(m, n);
        }
      }
    }
  }

  Eigen::Tensor<double, 4> tmpKT =
      tmpK.shuffle(Eigen::array<int, 4>{3, 2, 1, 0});

  Eigen::Tensor<double, 4> ints(2 * nBasis, 2 * nBasis, 2 * nBasis, 2 * nBasis);
  ints.setZero();
  for (int i = 0; i < 2 * nBasis; i++) {
    for (int j = 0; j < 2 * nBasis; j++) {
      Eigen::MatrixXd tmp1(nBasis, nBasis);
      tmp1.setZero();
      for (int k = 0; k < nBasis; k++) {
        for (int l = 0; l < nBasis; l++) {
          tmp1(k, l) = tmpKT(i, j, k, l);
        }
      }
      Eigen::MatrixXd kron = Eigen::kroneckerProduct(eye, tmp1).eval();
      for (int m = 0; m < 2 * nBasis; m++) {
        for (int n = 0; n < 2 * nBasis; n++) {
          ints(i, j, m, n) = kron(m, n);
        }
      }
    }
  }

  Eigen::MatrixXd CTotal(2 * nBasis, 2 * nBasis);
  CTotal.setZero();
  CTotal.block(0, 0, nBasis, nBasis) = C;
  CTotal.block(nBasis, nBasis, nBasis, nBasis) = C;

//   std::cout << "CTOTAL:\n" << CTotal << "\n";

  Eigen::Tensor<double, 2> CTensor = Eigen::TensorMap<Eigen::Tensor<double, 2>>(
      CTotal.data(), 2 * nBasis, 2 * nBasis);

  using IP = Eigen::IndexPair<int>;

  Eigen::Tensor<double, 4> tmp_ =
      ints.shuffle(Eigen::array<int, 4>{0, 2, 1, 3});
  Eigen::Tensor<double, 4> gao1 =
      tmp_ - tmp_.shuffle(Eigen::array<int, 4>{0, 1, 3, 2});

  Eigen::Tensor<double, 4> tmp1 =
      gao1.contract(CTensor, Eigen::array<IP, 1>{IP(3, 0)});
  Eigen::Tensor<double, 4> tmp2 =
      tmp1.contract(CTensor, Eigen::array<IP, 1>{IP(2, 0)});
  Eigen::Tensor<double, 4> tmp3 =
      tmp2.contract(CTensor, Eigen::array<IP, 1>{IP(1, 0)});
  Eigen::Tensor<double, 4> gmo =
      tmp3.contract(CTensor, Eigen::array<IP, 1>{IP(0, 0)});

  Eigen::Tensor<double, 2> H_core_ao = getAOKinetic() + getAOPotential();

  Eigen::MatrixXd H_core_ao_m =
      Eigen::Map<Eigen::MatrixXd>(H_core_ao.data(), nBasis, nBasis);

  Eigen::MatrixXd H_1body_ao(2 * nBasis, 2 * nBasis);
  H_1body_ao.block(0, 0, nBasis, nBasis) = H_core_ao_m;
  H_1body_ao.block(nBasis, nBasis, nBasis, nBasis) = H_core_ao_m;

  Eigen::Tensor<double, 2> H_1body_ao_Tensor =
      Eigen::TensorMap<Eigen::Tensor<double, 2>>(H_1body_ao.data(), 2 * nBasis,
                                                 2 * nBasis);

  // back to Tensor
  Eigen::Tensor<double, 2> H_1body(2 * nBasis, 2 * nBasis);
  H_1body.setZero();
  Eigen::Tensor<double, 2> tmpTensor =
      H_1body_ao_Tensor.contract(CTensor, Eigen::array<IP, 1>{IP(1, 0)});
  Eigen::Tensor<double, 2> CTensorT =
      CTensor.shuffle(Eigen::array<int, 2>{1, 0});
  Eigen::Tensor<double, 2> H_1body_Final =
      CTensorT.contract(tmpTensor, Eigen::array<IP, 1>{IP(1, 0)});


//   int c = 0;
//   for (int i = 0; i < 4; i++) {
//     for (int j = 0; j < 4; j++) {
//       for (int k = 0; k < 4; k++) {
//         for (int l = 0; l < 4; l++) {
//             if (std::fabs(gmo(i,j,k,l)) > 1e-12) {
//           std::cout << c << ", " << (std::fabs(gmo(i, j, k, l)) > 1e-10 ? gmo(i, j, k, l)
//                                                            : 0.0)
//                     << "\n";
//                     c++;
//             }
//         }
//       }
//     }
//   }


  std::vector<int> active_list(2*nBasis), frozen_list;
  std::iota(active_list.begin(), active_list.end(),0);
  int nActive = active_list.size();
  int nFrozen = 0;

  auto tmp_enuc = enuc;
  for (int i = 0; i < nFrozen; i++) {
      auto ia = frozen_list[i];
      tmp_enuc += H_1body_Final(ia,ia);
      for (int b = 0; b < i; b++) {
          auto ib = frozen_list[b];
          tmp_enuc += gmo(ia,ib,ia,ib);
      }
  }

  Eigen::Tensor<double, 2> h_fc_1body(nActive,nActive); h_fc_1body.setZero();
  for (int p = 0; p < nActive; p++ ){
      auto ip = active_list[p];
      for (int q = 0; q < nActive; q++ ) {
          auto iq = active_list[q];
          h_fc_1body(p,q) = H_1body_Final(ip,iq);

          for (int a = 0; a < nFrozen; a++) {
              auto ia = frozen_list[a];
              h_fc_1body(p,q) += gmo(ia,ip,ia,iq);
          }
      }
  }

  Eigen::Tensor<double, 4> h_fc_2body(nActive,nActive,nActive,nActive); h_fc_2body.setZero();
  for (int p = 0; p < nActive; p++) {
      auto ip = active_list[p];
      for (int q = 0; q < nActive; q++) {
          auto iq = active_list[q];
          for (int r = 0; r < nActive; r++) {
              auto ir = active_list[r];
              for (int ss = 0; ss < nActive; ss++) {
                  auto iss = active_list[ss];
                  h_fc_2body(p,q,r,ss) = gmo(ip,iq,ir,iss);
              }
          }
      }
  }

  Eigen::Tensor<double, 4> tmp_shuffle = h_fc_2body.shuffle(Eigen::array<int,4>{0,1,3,2});
  Eigen::Tensor<double, 4> h_fc_2body_tmp = tmp_shuffle * 0.25;

// std::cout << "One BOdy:\n"
//             << Eigen::Map<Eigen::MatrixXd>(h_fc_1body.data(), 2 * nBasis,
//                                            2 * nBasis)
//             << "\n";
  _hpq = h_fc_1body;//H_1body_Final;
  _hpqrs = h_fc_2body_tmp;
  _e_nuc = tmp_enuc;
}

Eigen::Tensor<double, 2> LibIntWrapper::hpq() { return _hpq; }
Eigen::Tensor<double, 4> LibIntWrapper::hpqrs() { return _hpqrs; }
const double LibIntWrapper::E_nuclear() { return _e_nuc; }
Eigen::Tensor<double, 2> LibIntWrapper::getAOKinetic() {
  return Eigen::TensorMap<Eigen::Tensor<double, 2>>(kinetic_data.data(), nBasis,
                                                    nBasis);
}

Eigen::Tensor<double, 2> LibIntWrapper::getAOPotential() {
  return Eigen::TensorMap<Eigen::Tensor<double, 2>>(potential_data.data(),
                                                    nBasis, nBasis);
}
Eigen::Tensor<double, 4> LibIntWrapper::getERI() {
  return Eigen::TensorMap<Eigen::Tensor<double, 4>>(eri_data.data(), nBasis,
                                                    nBasis, nBasis, nBasis);
}

} // namespace libintwrapper
