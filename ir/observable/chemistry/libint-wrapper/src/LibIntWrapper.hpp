#ifndef LIBINT_WRAPPER_HPP_
#define LIBINT_WRAPPER_HPP_

#include <map>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

namespace libintwrapper {
class LibIntWrapper {
public:
  void generate(std::map<std::string, std::string> &&options, std::vector<int> active = {}, std::vector<int> frozen = {});
  void generate(std::map<std::string, std::string> &options, std::vector<int> active = {}, std::vector<int> frozen = {});
  Eigen::Tensor<double, 2> getAOKinetic();
  Eigen::Tensor<double, 2> getAOPotential();
  Eigen::Tensor<double, 4> getERI();
  Eigen::Tensor<double, 2> hpq();
  Eigen::Tensor<double, 4> hpqrs();
  const double E_nuclear();

protected:
  std::vector<double> eri_data;
  std::vector<double> kinetic_data;
  std::vector<double> potential_data;
  Eigen::Tensor<double, 2> _hpq;
  Eigen::Tensor<double, 4> _hpqrs;
  double _e_nuc;

  int nBasis;

};

} // namespace libintwrapper
#endif