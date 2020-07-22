#ifndef EM_EXPLORATION_UTILS_H
#define EM_EXPLORATION_UTILS_H

#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#define DEG2RAD(x) ((x)*0.01745329251994329575)
#define RAD2DEG(x) ((x)*57.29577951308232087721)

namespace em_exploration {

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

template<typename Derived>
void saveToCSV(std::string name, const Eigen::DenseBase<Derived> &m) {
  std::ofstream file(name);
  if (!file.is_open()) {
    std::cerr << "File " << name << " can't be opened." << std::endl;
    return;
  }

  file << m.format(CSVFormat);
}

template<typename Scalar, int N>
Eigen::Matrix<Scalar, N, N> inverse(const Eigen::Matrix<Scalar, N, N> &m) {
  assert(m.rows() == m.cols());
  return m.llt().solve(Eigen::Matrix<Scalar, N, N>::Identity());
}

double maxEigenvalue(const Eigen::MatrixXd &m);

double minEigenvalue(const Eigen::MatrixXd &m);

}
#endif //EM_EXPLORATION_UTILS_H
