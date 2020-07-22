#include "em_exploration/Utils.h"

namespace em_exploration {

double maxEigenvalue(const Eigen::MatrixXd &m) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
  return es.eigenvalues()[m.rows() - 1];
}

double minEigenvalue(const Eigen::MatrixXd &m) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
  return es.eigenvalues()[0];
}

}
