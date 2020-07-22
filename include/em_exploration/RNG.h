#ifndef EM_EXPLORATION_RNG_H
#define EM_EXPLORATION_RNG_H

#include <random>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "halton/halton.hpp"

namespace em_exploration {

/**
 * Quasi-random number generator using Halton sequence.
 */
class QRNG {
 public:
  QRNG(int dim, int count=0) : dim_(dim), count_(count) {}

  void setDim(int dim) {
    dim_ = dim;
  }

  void setCount(int count) {
    count_ = count;
  }

  int getCount() const { return count_; }

  Eigen::VectorXd generate() {
    double *values = halton(count_++, dim_);

    Eigen::VectorXd v(dim_);
    for (int i = 0; i < dim_; ++i)
      v(i) = values[i];
    delete[] values;
    return v;
  }

 private:
  int count_;
  int dim_;
};

/**
 * Random number generators.
 */
class RNG {
 public:
  typedef std::mt19937::result_type SeedType;
  RNG() :
      generator_(time(0)),
      uniform_real_dist_(0.0, 1.0),
      normal_dist_(0.0, 1.0) {}

  RNG(SeedType seed) :
      generator_(seed),
      uniform_real_dist_(0.0, 1.0),
      normal_dist_(0.0, 1.0) {}

  void setSeed(SeedType seed) {
    generator_.seed(seed);
  }

  double uniform01() {
    return uniform_real_dist_(generator_);
  }

  double uniformReal(double low, double high) {
    assert(high >= low);
    return (high - low) * uniform_real_dist_(generator_) + low;
  }

  int uniformInt(int low, int high) {
    assert(high >= low);
    int n = (int) floor(uniformReal((double) low, high + 1.0));
    return n >= low ? (n <= high ? n : high) : low;
  }

  bool uniformBool() {
    return uniform_real_dist_(generator_) > 0.5;
  }

  double normal01() {
    return normal_dist_(generator_);
  }

  double normal(double m, double std) {
//    double n;
//    while (1) {
//      n  = normal_dist_(generator_);
//      if (fabs(n) < 0.5)
//        break;
//    }
//    return n * std + m;
    return normal_dist_(generator_) * std + m;
  }

  bool bernoulli(double p) {
    return uniform01() < p;
  }

  void setMultivariateNormal(const Eigen::VectorXd &m, const Eigen::MatrixXd &cov) {
    assert(m.rows() == cov.rows());
    Eigen::LLT<Eigen::MatrixXd> solver(cov);
    assert(solver.info() == Eigen::Success);

    m_ = m;
    L_ = solver.matrixL();
  }

  Eigen::VectorXd multivariateNormal() {
    Eigen::VectorXd u(m_.size());
    for (int i = 0; i < m_.size(); ++i)
      u(i) = normal01();

    return L_ * u + m_;
  };

 private:
  std::mt19937 generator_;
  std::uniform_real_distribution<> uniform_real_dist_;
  std::normal_distribution<> normal_dist_;

  Eigen::MatrixXd L_;
  Eigen::VectorXd m_;
};
}

#endif //EM_EXPLORATION_RNG_H
