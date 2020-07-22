#ifndef EM_EXPLORATION_SLAM2D_H
#define EM_EXPLORATION_SLAM2D_H

#include <fstream>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>

#include "em_exploration/Simulation2D.h"
#include "em_exploration/RNG.h"
#include "em_exploration/FastMarginals.h"

#define USE_FAST_MARGINAL

#include <Eigen/SparseCore>
#include <boost/unordered_map.hpp>
#include <unordered_map>

namespace em_exploration {
class SLAM2D {

 public:
  typedef gtsam::BearingRangeFactor<Pose2, Point2> MeasurementFactor2D;
  typedef boost::shared_ptr<MeasurementFactor2D> MeasurementFactor2DPtr;
  typedef gtsam::BetweenFactor<Pose2> OdometryFactor2D;
  typedef boost::shared_ptr<gtsam::BetweenFactor<gtsam::Pose2>> OdometryFactor2DPtr;

  SLAM2D(const Map::Parameter &parameter, RNG::SeedType seed = 0);

  ~SLAM2D() {}

  void printParameters() const;

  void fromISAM2(std::shared_ptr<gtsam::ISAM2> isam, const Map &map, const gtsam::Values values);

  void addPrior(unsigned int key, const LandmarkBeliefState &landmark_state);

  void addPrior(const VehicleBeliefState &vehicle_state);

  static OdometryFactor2DPtr buildOdometryFactor(unsigned int x1, unsigned int x2,
                                                 const SimpleControlModel::ControlState &control_state);

  void addOdometry(const SimpleControlModel::ControlState &control_state);

  static MeasurementFactor2DPtr buildMeasurementFactor(unsigned int x, unsigned int l,
                                                       const BearingRangeSensorModel::Measurement &measurement);

  void addMeasurement(unsigned int key, const BearingRangeSensorModel::Measurement &measurement);

  /// Save graph to .dot file
  /// Visualize the graph with `dot -Tpdf ./graph.dot -O`"
  void saveGraph(const std::string &name = std::string("graph.dot")) const;

  void printGraph() const;

  std::vector<gtsam::Key> get_all_key() const;

  int key_size() const;
  double goal_x(gtsam::Key) const;
  double goal_y(gtsam::Key) const;

  void adjacency_degree_get();

  std::pair<double, double> get_key_points(int key);

  Eigen::MatrixXd adjacency_out(){
      return adjacency_matrix_;
  }

  Eigen::MatrixXd features_out(){
    return features_matrix_;
  }

  Eigen::MatrixXd jointMarginalCovarianceLocal(const std::vector<unsigned int> &poses,
                                          const std::vector<unsigned int> &landmarks) const;

  Eigen::MatrixXd jointMarginalCovariance(const std::vector<unsigned int> &poses,
                                  const std::vector<unsigned int> &landmarks) const;

  std::shared_ptr<const gtsam::ISAM2> getISAM2() const;

  /// Perform optimization and update the map including best estimate.
  /// The covariance of the lastest pose is updated if update_covariacne is true.
  void optimize(bool update_covariance = true);

  void copy_optimize(bool update_covariance = true);
  void set_copy_isam();

  /// Sample map (trajectory and landmarks) from posteriors
  std::pair<double, std::shared_ptr<Map>> sample() const;

  const Map &getMap() const { return map_; }

  static gtsam::Symbol getVehicleSymbol(unsigned int key) {
    return gtsam::Symbol('x', key);
  }

  static gtsam::Symbol getLandmarkSymbol(unsigned int key) {
    return gtsam::Symbol('l', key);
  }

#ifdef USE_FAST_MARGINAL
  std::shared_ptr<FastMarginals> getMarginals() const {
    return marginals_;
  }
#endif

  Eigen::MatrixXd adjacency_matrix_;
  Eigen::MatrixXd features_matrix_;

 private:
  double optimizeInPlacePerturbation(const gtsam::ISAM2Clique::shared_ptr &clique,
                                   gtsam::VectorValues &result) const;

  Map map_;
  unsigned int step_;

  std::shared_ptr<gtsam::ISAM2> isam_;
  std::shared_ptr<gtsam::ISAM2> copy_isam_;


  mutable bool marginals_update_;
#ifdef USE_FAST_MARGINAL
  mutable std::shared_ptr<FastMarginals> marginals_;
#else
  mutable std::shared_ptr<gtsam::Marginals> marginals_;
#endif

  gtsam::NonlinearFactorGraph graph_;
  gtsam::NonlinearFactorGraph graph_all_;
  gtsam::Values initial_estimate_;
  gtsam::Values result_;
  bool optimized_;

  mutable RNG rng_;
};
}
#endif //EM_EXPLORATION_SLAM2D_H
