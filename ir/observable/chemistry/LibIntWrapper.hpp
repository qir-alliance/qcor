#ifndef QCOR_IR_CHEMISTRY_OBSERVABLE_LIBINT_HPP_
#define QCOR_IR_CHEMISTRY_OBSERVABLE_LIBINT_HPP_

#include <unordered_map>
#include <vector>
#include <thread>

#include <Eigen/Dense>

#include <libint2.hpp>
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using libint2::Shell;
using libint2::Atom;
using libint2::BasisSet;
using libint2::Operator;
using libint2::BraKet;

const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;


namespace libint2 {
int nthreads = 1;

/// fires off \c nthreads instances of lambda in parallel
template <typename Lambda>
void parallel_do(Lambda& lambda) {
#ifdef _OPENMP
#pragma omp parallel
  {
    auto thread_id = omp_get_thread_num();
    lambda(thread_id);
  }
#else  // use C++11 threads
  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id != libint2::nthreads; ++thread_id) {
    if (thread_id != nthreads - 1)
      threads.push_back(std::thread(lambda, thread_id));
    else
      lambda(thread_id);
  }  // threads_id
  for (int thread_id = 0; thread_id < nthreads - 1; ++thread_id)
    threads[thread_id].join();
#endif
}
}

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
shellpair_list_t obs_shellpair_list;  // shellpair list for OBS
using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;  // in same order as shellpair_list_t
shellpair_data_t obs_shellpair_data;  // shellpair data for OBS

std::tuple<shellpair_list_t,shellpair_data_t>
compute_shellpairs(const BasisSet& bs1,
                   const BasisSet& _bs2 = BasisSet(),
                   double threshold = 1e-12){
  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  using libint2::nthreads;

  // construct the 2-electron repulsion integrals engine
  using libint2::Engine;
  std::vector<Engine> engines;
  engines.reserve(nthreads);
  engines.emplace_back(Operator::overlap,
                       std::max(bs1.max_nprim(), bs2.max_nprim()),
                       std::max(bs1.max_l(), bs2.max_l()), 0);
  for (size_t i = 1; i != nthreads; ++i) {
    engines.push_back(engines[0]);
  }

  std::cout << "computing non-negligible shell-pair list ... ";

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

  shellpair_list_t splist;

  std::mutex mx;

  auto compute = [&](int thread_id) {

    auto& engine = engines[thread_id];
    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      mx.lock();
      if (splist.find(s1) == splist.end())
        splist.insert(std::make_pair(s1, std::vector<size_t>()));
      mx.unlock();

      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        if (s12 % nthreads != thread_id) continue;

        auto on_same_center = (bs1[s1].O == bs2[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          auto n2 = bs2[s2].size();
          engines[thread_id].compute(bs1[s1], bs2[s2]);
          Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
          auto norm = buf_mat.norm();
          significant = (norm >= threshold);
        }

        if (significant) {
          mx.lock();
          splist[s1].emplace_back(s2);
          mx.unlock();
        }
      }
    }
  };  // end of compute

  libint2::parallel_do(compute);

  // resort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2] if s1 < s2
  // N.B. only parallelized over 1 shell index
  auto sort = [&](int thread_id) {
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        auto& list = splist[s1];
        std::sort(list.begin(), list.end());
      }
    }
  };  // end of sort

  libint2::parallel_do(sort);

  // compute shellpair data assuming that we are computing to default_epsilon
  // N.B. only parallelized over 1 shell index
  const auto ln_max_engine_precision = std::log(max_engine_precision);
  shellpair_data_t spdata(splist.size());
  auto make_spdata = [&](int thread_id) {
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        for(const auto& s2 : splist[s1]) {
          spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(bs1[s1],bs2[s2],ln_max_engine_precision));
        }
      }
    }
  };  // end of make_spdata

  libint2::parallel_do(make_spdata);

  timer.stop(0);
  std::cout << "done (" << timer.read(0) << " s)" << std::endl;

  return std::make_tuple(splist,spdata);
}


template <Operator obtype, typename OperatorParams = typename libint2::operator_traits<obtype>::oper_params_type>
std::array<Matrix, libint2::operator_traits<obtype>::nopers> compute_1body_ints(
    const BasisSet& obs,
    OperatorParams oparams =
        OperatorParams()) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  using libint2::nthreads;
  typedef std::array<Matrix, libint2::operator_traits<obtype>::nopers>
      result_type;
  const unsigned int nopers = libint2::operator_traits<obtype>::nopers;
  result_type result;
  for (auto& r : result) r = Matrix::Zero(n, n);

  // construct the 1-body integrals engine
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] = libint2::Engine(obtype, obs.max_nprim(), obs.max_l(), 0);
  // pass operator params to the engine, e.g.
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical
  // charges
  engines[0].set_params(oparams);
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto shell2bf = obs.shell2bf();

  auto compute = [&](int thread_id) {

    const auto& buf = engines[thread_id].results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over
    // Hermitian operators: (1|2) = (2|1)
    for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();

      auto s1_offset = s1 * (s1+1) / 2;
      for (auto s2: obs_shellpair_list[s1]) {
        auto s12 = s1_offset + s2;
        if (s12 % nthreads != thread_id) continue;

        auto bf2 = shell2bf[s2];
        auto n2 = obs[s2].size();

        auto n12 = n1 * n2;

        // compute shell pair; return is the pointer to the buffer
        engines[thread_id].compute(obs[s1], obs[s2]);

        for (unsigned int op = 0; op != nopers; ++op) {
          // "map" buffer to a const Eigen Matrix, and copy it to the
          // corresponding blocks of the result
          Eigen::Map<const Matrix> buf_mat(buf[op], n1, n2);
          result[op].block(bf1, bf2, n1, n2) = buf_mat;
          if (s1 != s2)  // if s1 >= s2, copy {s1,s2} to the corresponding
                         // {s2,s1} block, note the transpose!
            result[op].block(bf2, bf1, n2, n1) = buf_mat.transpose();
        }
      }
    }
  };  // compute lambda

  libint2::parallel_do(compute);

  return result;
}


#endif