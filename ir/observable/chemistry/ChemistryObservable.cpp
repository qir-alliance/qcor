#include "ChemistryObservable.hpp"
#include "FermionOperator.hpp"

#include "LibIntWrapper.hpp"

#include "xacc_service.hpp"
#include "ObservableTransform.hpp"

#include <iostream>

namespace qcor {
namespace observable {

std::vector<std::shared_ptr<Function>>
ChemistryObservable::observe(std::shared_ptr<Function> function) {
   auto jw = xacc::getService<xacc::ObservableTransform>("jw");
//    std::cout << "Fermion:\n" << fermionOp->toString() << "\n";
   auto pauli = jw->transform(fermionOp);
//    std::cout << "Pauli:\n" << pauli->toString() << "\n";
   return pauli->observe(function);
}

const std::string ChemistryObservable::toString() { return ""; }

void ChemistryObservable::fromString(const std::string str) {
  XACCLogger::instance()->error(
      "ChemisryObservable.fromString not implemented.");
}

const int ChemistryObservable::nBits() { return fermionOp->nBits(); }

void ChemistryObservable::fromOptions(
    std::map<std::string, InstructionParameter> &&options) {
  fromOptions(options);
}

void ChemistryObservable::fromOptions(
    std::map<std::string, InstructionParameter> &options) {

  std::map<std::string, std::string> opts;
  for (auto &kv : options) {
    opts.insert({kv.first, kv.second.toString()});
  }
  libintwrapper::LibIntWrapper libint;
  libint.generate(opts);

  auto hpq = libint.hpq();
  auto hpqrs = libint.hpqrs();
  auto enuc = libint.E_nuclear();

  std::stringstream ss;
  ss << enuc;
  for (int i = 0; i < hpq.dimension(0); i++) {
    for (int j = 0; j < hpq.dimension(1); j++) {
      if (std::fabs(hpq(i, j)) > 1e-12) {
        auto negOrPlus = hpq(i, j) < 0.0 ? " - " : " + ";
        ss << negOrPlus << std::fabs(hpq(i, j)) << " " << i << "^ " << j;
      }
    }
  }

  for (int i = 0; i < hpqrs.dimension(0); i++) {
    for (int j = 0; j < hpqrs.dimension(1); j++) {
      for (int k = 0; k < hpqrs.dimension(2); k++) {
        for (int l = 0; l < hpqrs.dimension(3); l++) {
          if (std::fabs(hpqrs(i, j, k, l)) > 1e-12) {
            auto negOrPlus = hpqrs(i, j, k, l) < 0.0 ? " - " : " + ";
            ss << negOrPlus << std::fabs(hpqrs(i, j, k, l)) << " " << i << "^ "
               << j << "^ " << k << " " << l;
          }
        }
      }
    }
  }

  fermionOp =
      std::make_shared<xacc::quantum::FermionOperator>(ss.str());
}

} // namespace observable
} // namespace qcor
