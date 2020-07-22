#ifndef EM_EXPLORATION_FASTMARGINALS_H
#define EM_EXPLORATION_FASTMARGINALS_H

#include <unordered_map>
#include <boost/unordered_map.hpp>

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>

namespace em_exploration {

/*
 * Implementation of the covariance recovery algorithm in
 *     Covariance Recovery from a Square Root Information Matrix for Data Association
 * See code in iSAM for details
 *     http://people.csail.mit.edu/kaess/isam/
 */
struct SparseBlockVector {
  std::map<gtsam::Key, gtsam::Matrix> vector;

  void insert(gtsam::Key i, const gtsam::Matrix &block) {
    vector[i] = block;
  }

  std::map<gtsam::Key, gtsam::Matrix>::const_iterator begin() const { return vector.begin(); };
  std::map<gtsam::Key, gtsam::Matrix>::const_iterator end() const { return vector.end(); };

  SparseBlockVector() {}
};

std::ostream& operator<<(std::ostream &os, const SparseBlockVector &vector);

typedef std::pair<gtsam::Key, gtsam::Key> KeyPair;
std::size_t hash_value(const KeyPair &key_pair);

struct CovarianceCache {
  boost::unordered_map<KeyPair, gtsam::Matrix> entries;
  std::unordered_map<gtsam::Key, gtsam::Matrix> diag;
  std::unordered_map<gtsam::Key, SparseBlockVector> rows;

  CovarianceCache() {}
};

class FastMarginals {
  friend class FastMarginals2;

 public:
  FastMarginals(const std::shared_ptr<gtsam::ISAM2> &isam2) : isam2_(isam2) {
    initialize();
  }

  gtsam::Matrix marginalCovariance(const gtsam::Key &variable);

  gtsam::Matrix jointMarginalCovariance(const std::vector<gtsam::Key> &variables);

 protected:

  void initialize();

  gtsam::Matrix getRBlock(const gtsam::Key &key_i, const gtsam::Key &key_j);

  const SparseBlockVector& getRRow(const gtsam::Key &key);

  gtsam::Matrix getR(const std::vector<gtsam::Key> &variables);

  gtsam::Matrix getKeyDiag(const gtsam::Key &key);

  size_t getKeyDim(const gtsam::Key &key);

  gtsam::Matrix sumJ(const gtsam::Key key_l, const gtsam::Key key_i);

  gtsam::Matrix recover(const gtsam::Key &key_i, const gtsam::Key &key_l);

  std::shared_ptr<gtsam::ISAM2> isam2_;
  std::vector<gtsam::Key> ordering_;
  boost::unordered_map<gtsam::Key, size_t> key_idx_;
  CovarianceCache cov_cache_;
  std::unordered_map<gtsam::Key, gtsam::Matrix> Fs_;
  std::unordered_map<gtsam::Key, gtsam::Matrix> F_;
  gtsam::Key last_key_;
  size_t size0_;
};

/*
 * Implementation of covariance update given predicted inputs without the need of relinearization
 * See Fig. 3 for details
 *     Fast covariance recovery in incremental nonlinear least squares solvers
 */
class FastMarginals2 : public FastMarginals {
 public:
  FastMarginals2(const std::shared_ptr<gtsam::ISAM2> &isam2) : FastMarginals(isam2) {}

  FastMarginals2(const std::shared_ptr<FastMarginals> &fast_marginals)
      : FastMarginals(*fast_marginals), fast_marginals_(fast_marginals) {}

  void update(const gtsam::NonlinearFactorGraph &odom_graph,
              const gtsam::NonlinearFactorGraph &meas_graph,
              const gtsam::Values &values,
              const gtsam::KeySet &updated_keys);
 private:
  gtsam::Matrix propagate(gtsam::Key key0, gtsam::Key key1);

  gtsam::KeySet new_keys_;
  std::unordered_map<gtsam::Key, gtsam::JacobianFactor::shared_ptr> linear_odom_factors_;

  std::shared_ptr<FastMarginals> fast_marginals_;
};
}

#endif //EM_EXPLORATION_FASTMARGINALS_H
