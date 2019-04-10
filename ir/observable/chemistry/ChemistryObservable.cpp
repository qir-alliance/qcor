#include "ChemistryObservable.hpp"
#include "FermionOperator.hpp"

#include <iostream>

namespace qcor {
namespace observable {

namespace py = pybind11;
using namespace py::literals;

std::vector<std::shared_ptr<Function>>
ChemistryObservable::observe(std::shared_ptr<Function> function) {}

const std::string ChemistryObservable::toString() { return ""; }

void ChemistryObservable::fromString(const std::string str) {
  XACCLogger::instance()->error(
      "ChemisryObservable.fromString not implemented.");
}

const int ChemistryObservable::nBits() { return 0; }


void ChemistryObservable::fromOptions(
    std::map<std::string, InstructionParameter> &&options) {
  fromOptions(options);
}


void ChemistryObservable::fromOptions(
    std::map<std::string, InstructionParameter> &options) {}

} // namespace observable
} // namespace qcor


// void ChemistryObservable::fromOptions(
//     std::map<std::string, InstructionParameter> &&options) {

//   auto geom = options["geometry"].toString();
//   std::ofstream tmp(".chem_obs_geom.xyz");
//   tmp << geom;
//   tmp.close();
//   std::ifstream input_file(".chem_obs_geom.xyz");

//   auto atoms = libint2::read_dotxyz(input_file);

//   // count the number of electrons
//   auto nelectron = 0;
//   for (auto i = 0; i < atoms.size(); ++i)
//     nelectron += atoms[i].atomic_number;
//   const auto ndocc = nelectron / 2;
//   std::cout << "# of electrons = " << nelectron << std::endl;

//   auto enuc = 0.0;
//   for (auto i = 0; i < atoms.size(); i++) {
//     for (auto j = i + 1; j < atoms.size(); j++) {
//       auto xij = atoms[i].x - atoms[j].x;
//       auto yij = atoms[i].y - atoms[j].y;
//       auto zij = atoms[i].z - atoms[j].z;
//       auto r2 = xij * xij + yij * yij + zij * zij;
//       auto r = std::sqrt(r2);
//       enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
//     }
//   }
//   std::cout << "Nuclear repulsion energy = " << std::setprecision(15) << enuc
//             << std::endl;

//   libint2::Shell::do_enforce_unit_normalization(false);

//   //   auto molecule = std::make_shared<psi::Molecule>();

//   std::cout << "Atomic Cartesian coordinates (a.u.):" << std::endl;
//   for (const auto &a : atoms) {
//     // molecule->add_atom(a.atomic_number, a.x, a.y, a.z);

//     std::cout << a.atomic_number << " " << a.x << " " << a.y << " " << a.z
//               << std::endl;
//   }

//   //   using namespace psi;

//   libint2::initialize();
//   //   std::shared_ptr<BasisSetParser> parser(new Gaussian94BasisSetParser());
//   //  std::shared_ptr<BasisSet> aoBasis = BasisSet::construct(parser, molecule,
//   //  options["basis"].toString());
//   libint2::BasisSet obs(options["basis"].toString(), atoms, true);
//   std::copy(begin(obs), end(obs),
//             std::ostream_iterator<libint2::Shell>(std::cout, "\n"));

//   libint2::Engine eri_engine(libint2::Operator::coulomb, obs.max_nprim(),
//                              obs.max_l(), 0);
//   //   eri_engine.set(libint2::BraKet::xs_xs);

//   auto shell2bf = obs.shell2bf(); // maps shell index to basis function index
//                                   // shell2bf[0] = index of the first basis
//                                   // function in shell 0 shell2bf[1] = index of
//                                   // the first basis function in shell 1
//                                   // ...
//   const auto &buf = eri_engine.results(); // will point to computed shell sets

//   // const auto& is very important!
//   auto n = obs.nbf();
//   auto nshells = obs.size();
//   Matrix result = Matrix::Zero(n, n);

//   //  for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
//   //       auto bf1 = shell2bf[s1];  // first basis function in this shell
//   //       auto n1 = obs[s1].size();

//   //       for (auto s2 = 0; s2 <= s1; ++s2, ++s12) {

//   //         auto bf2 = shell2bf[s2];
//   //         auto n2 = obs[s2].size();

//   //         // compute shell pair; return is the pointer to the buffer
//   //         eri_engine.compute(obs[s1], obs[s2], obs[s1], obs[s2]);
//   //         if (buf[0] == nullptr)
//   //           continue; // if all integrals screened out, skip to next shell
//   //           set

//   //         // "map" buffer to a const Eigen Matrix, and copy it to the
//   //         // corresponding blocks of the result
//   //         Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
//   //         result.block(bf1, bf2, n1, n2) = buf_mat;
//   //         if (s1 != s2)  // if s1 >= s2, copy {s1,s2} to the corresponding
//   //         {s2,s1}
//   //                        // block, note the transpose!
//   //           result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
//   //       }
//   //     }

//   Eigen::Tensor<double,1> data_(16);

//   std::vector<double> data;
// //   std::vector<double> data;
//   int ii = 0;
//   //   std::cout << "MATRIX:\n" << result << "\n";
//   for (auto s1 = 0; s1 != obs.size(); ++s1) {
//     for (auto s2 = 0; s2 != obs.size(); ++s2) {
//       for (auto s3 = 0; s3 != obs.size(); ++s3) {
//         for (auto s4 = 0; s4 != obs.size(); ++s4) {

//           std::cout << "compute shell set {" << s1 << "," << s2 << "," << s3
//                     << "," << s4 << "} ... ";
//           eri_engine.compute(obs[s1], obs[s2], obs[s3], obs[s4]);
//           std::cout << "done. ";
//           auto ints_shellset = buf[0]; // location of the computed integrals
//           if (ints_shellset == nullptr)
//             continue; // nullptr returned if the entire shell-set was screened
//                       // out

//           auto bf1 = shell2bf[s1];  // first basis function in first shell
//           auto n1 = obs[s1].size(); // number of basis functions in first shell
//           auto bf2 = shell2bf[s2];  // first basis function in second shell
//           auto n2 = obs[s2].size(); // number of basis functions in second shell
//           auto bf3 = shell2bf[s3];  // first basis function in first shell
//           auto n3 = obs[s3].size(); // number of basis functions in first shell
//           auto bf4 = shell2bf[s4];  // first basis function in second shell
//           auto n4 = obs[s4].size(); // number of basis functions in second shell

//           // integrals are packed into ints_shellset in row-major (C) form
//           // this iterates over integrals in this order
//         //   for (int ii = 0;
//             //    ii < sizeof(ints_shellset) / sizeof(ints_shellset[0]); ii++) {
//             std::cout << ii << ", " << ints_shellset[0] << "\n";
//             data.push_back(ints_shellset[0]);
//             data_(ii) = ints_shellset[0];
//         //   }
//         ii++;
//         }
//       }
//     }
//   }

// //   std::cout << "EIgen:\n" << data << "\n";
//   std::vector<int> shape{2,2,2,2};
//   std::vector<double> idvec {1.,0.,0.,1.};

//  auto arr = xt::adapt(data, shape);
//  xt::xarray<double> darr = arr;

//  auto identity = xt::adapt(idvec, std::vector<int>{1,1,2,2});
//  xt::xarray<double> did = identity;
//  auto tmp1 = xt::linalg::kron(did, darr);

//   auto && sh1 = tmp1.shape();
//   for (auto& el : sh1) {std::cout << el << ", "; }

//   std::cout << "\n";
// //  auto kron2 = xt::linalg::kron(identity, xt::transpose(kron, {3,2,1,0}));
// //  std::cout << arr << "\n" << identity << "\n";
// //  auto && sh = kron2.shape();
// // for (auto& el : sh) {std::cout << el << ", "; }
// // std::cout << "\n";
//   std::array<int, 4> dims{2,2,2,2};

//   Eigen::Tensor<double, 4> data4 = data_.reshape(dims);


//  Eigen::Tensor<double, 2> identityT(2,2);
//  identityT.setZero();
//  for (int i = 0; i < 2; i++) identityT(i,i) = 1.0;

// Eigen::Tensor<double,4> idt = identityT.reshape(std::array<int,4>{1,1,2,2});

// Eigen::Tensor<double,8> kroned = idt.contract(data4, std::array<Eigen::IndexPair<int>,0>{});

//  for (int i = 0; i < 8; i++) std::cout << kroned.dimension(i) << " ";
//  std::cout << "\n";

//  std::vector<double> dd(kroned.data(), kroned.data() + kroned.size());

//  auto adapted = xt::adapt(dd, std::vector<int>{1,1,2,2,2,2,2,2});

//  std::vector<std::vector<int>> v{ {1,2,2,2,2,2,2}, {2,2,2,2,2,2}, {2,2,2,4,2}, {2,2,4,4} };
//  for (int i = 0; i < 4; i++) {
//      adapted = xt::concatenate(xt::xtuple(adapted, xt::ones<double>({1,1,2,2,2,2,2,2})), 3);
//      adapted.reshape(v[i]);
//     auto && sh11 = adapted.shape();
//   for (auto& el : sh11) {std::cout << el << ", "; }

//   std::cout << "\n";
//  }

//  auto && sh11 = adapted.shape();
//   for (auto& el : sh11) {std::cout << el << ", "; }

//   std::cout << "\n";

// //  Eigen::Tensor<double,7> tmp1(1,2,2,2,2,2,2);
// //  Eigen::Tensor<double,7> tmp2 = kroned.concatenate(tmp1,3);
// //  Eigen::Tensor<double,6> tmp2 = tmp1.concatenate(tmp1,3);
// //  Eigen::Tensor<double,5> tmp3 = tmp2.concatenate(tmp2,3);
// //  Eigen::Tensor<double,4> tmp4 = tmp3.concatenate(tmp3,3);

// //  for (int i = 0; i < 4; i++) std::cout << tmp4.dimension(i) << " ";

// //  std::cout << "\n";
// // [[[0.77460594 0.44037993]
// //    [0.44037993 0.56721469]]

// //   [[0.44037993 0.2927097 ]
// //    [0.2927097  0.44037993]]]


// //  [[[0.44037993 0.2927097 ]
// //    [0.2927097  0.44037993]]

// //   [[0.56721469 0.44037993]
// //    [0.44037993 0.77460594]]]]

//   //   std::cout << "Max nprim " << obs.max_nprim() << "\n";
//   //   std::cout << "orbital basis set rank = " << obs.nbf() << std::endl;

//   //   libint2::initialize();

//   //    {
//   //       std::tie(obs_shellpair_list, obs_shellpair_data) =
//   //       compute_shellpairs(obs); size_t nsp = 0; for (auto& sp :
//   //       obs_shellpair_list) {
//   //         nsp += sp.second.size();
//   //       }
//   //       std::cout << "# of {all,non-negligible} shell-pairs = {"
//   //                 << obs.size() * (obs.size() + 1) / 2 << "," << nsp <<
//   //                 "}"
//   //                 << std::endl;
//   //     }

//   //    auto S = compute_1body_ints<Operator::overlap>(obs)[0];

//   //    auto T = compute_1body_ints<Operator::kinetic>(obs)[0];
//   //     auto V = compute_1body_ints<Operator::nuclear>(obs,
//   //     libint2::make_point_charges(atoms))[0]; Matrix H = T + V;
//   //     T.resize(0, 0);
//   //     V.resize(0, 0);

//   //     std::cout << "Matrix:\n" << H << "\n";

//   //  std::cout << "S:\n" << S << "\n";

//   //   libint2::finalize();
// }


//   std::shared_ptr<py::scoped_interpreter> guard;
//   guard = std::make_shared<py::scoped_interpreter>();

// //   py::print("quil:\n", quilStr);
//   auto np = py::module::import("numpy");
//   auto psi4 = py::module::import("psi4");

//   auto moleculeGeom = psi4.attr("geometry")(options["geometry"].toString());
//   psi4.attr("set_options")(py::dict("basis"_a=options["basis"].toString()));
//   auto e_and_wfn = psi4.attr("energy")("scf", "return_wfn"_a=true).cast<py::tuple>();
//   std::cout << "made it here\n";
//   std::cout << "Energy " << py::cast<double>(e_and_wfn[0]) << "\n";

//   auto mints = psi4.attr("core").attr("MintsHelper")(e_and_wfn[1].attr("basisset")());
//   auto nbf = mints.attr("nbf")().cast<int>();
//   auto nso = 2 * nbf;
//   auto nalpha = e_and_wfn[1].attr("nalpha")().cast<int>();
//   auto nbeta = e_and_wfn[1].attr("nbeta")().cast<int>();
//   auto nocc = nalpha + nbeta;
//   auto nvirt = 2 * nbf - nocc;
//   auto list_occ_alpha = np.attr("asarray")(e_and_wfn[1].attr("occupation_a")());
//   auto list_occ_beta = np.attr("asarray")(e_and_wfn[1].attr("occupation_a")());

//   std::cout << nbf << ", " << nvirt << "\n";

//   auto eps_a = np.attr("asarray")(e_and_wfn[1].attr("epsilon_a")());
//   auto eps_b = np.attr("asarray")(e_and_wfn[1].attr("epsilon_b")());
//   auto eps = np.attr("append")(eps_a, eps_b);

//   auto Ca = np.attr("asarray")(e_and_wfn[1].attr("Ca")());
//   auto Cb = np.attr("asarray")(e_and_wfn[1].attr("Cb")());

//   py::list l1(2);
//   py::list l2(2);
//   py::list l3(2);

//   l2[0] = Ca;
//   l2[1] = np.attr("zeros_like")(Cb);
//   l3[0] = np.attr("zeros_like")(Ca);
//   l3[1] = Cb;

//   l1[0] = l2;
//   l1[1] = l3;
//   auto C = np.attr("block")(l1);

//   auto ints = np.attr("asarray")(mints.attr("ao_eri")());
//   auto identity = np.attr("eye")(2);
//   ints = np.attr("kron")(identity,ints);
//   ints = np.attr("kron")(identity, ints.attr("T"));

//   auto tmp = ints.attr("transpose")(0,2,1,3);
//   auto gao = tmp.attr("__sub__")(tmp.attr("transpose")(0,1,3,2));
//   py::print(gao.attr("shape"));
//   py::print(C.attr("shape"));

//   auto tmp1 = np.attr("einsum")("pqrs, sS -> pqrS", gao, C);
//   py::print(gao.attr("shape"));
// //   auto tmp2 = np.attr("einsum")("pqrS, rR -> pqRS", tmp1, C);
// //   auto tmp3 = np.attr("einsum")("pqRS, qQ -> pQRS", tmp2, C);
// //   auto gmo = np.attr("einsum")("pQRS, pP -> PQRS", tmp3, C);



// //   auto bs = e_and_wfn[1].attr("basisset")();
// //   py::object get_qc = pyquil.attr("get_qc");
// //   auto program = pyquil.attr("Program")();
// //   program.attr("inst")(quilStr);
// //   program.attr("wrap_in_numshots_loop")(shots);


