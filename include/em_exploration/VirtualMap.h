#ifndef EM_EXPLORATION_VIRTUALMAP_H
#define EM_EXPLORATION_VIRTUALMAP_H

#include "em_exploration/RNG.h"
#include "em_exploration/Utils.h"
#include "em_exploration/Simulation2D.h"
#include "em_exploration/SLAM2D.h"
#include "em_exploration/OccupancyMap.h"

namespace em_exploration {

/**
 * The virtual map consists of virtual landmarks.
 * The uncertainty of virtual landmarks needs to be reduced during exploration.
 * Virtual landmarks could be actual landmarks in the map, and the probability is
 * computed by expected occupancy grid maps.
 */
class VirtualMap {
 public:

  class Parameter : public Map::Parameter {
   public:
    Parameter(const Map::Parameter &map_parameter);

    inline double getResolution() const { return resolution_; }
    inline double getSigma0() const { return sigma0_; }
    inline double getNumSamples() const { return num_samples_; }
    inline void setResolution(double value) { resolution_ = value; }
    inline void setSigma0(double value) { sigma0_ = value; }
    inline void setNumSamples(int value) { num_samples_ = value; }

    void print() const;

   private:
    double resolution_;
    double sigma0_;  /// std. dev. of virtual landmarks
    int num_samples_;  /// number of trajectory samples
  };

  struct VirtualLandmark {
    bool updated;   /// the covariance has been updated at least once
    double probability;   /// the probability of being actual landmark

    Point2 point;
    Eigen::Matrix2d information;

    VirtualLandmark() {}
    VirtualLandmark(double probability, const Point2 &point, const Eigen::Matrix2d &information)
        : updated(false), probability(probability), point(point), information(information) {}

    Eigen::Matrix2d covariance() const {
      return inverse(information);
    }
  };

  VirtualMap(const Parameter &parameter);

  VirtualMap(const Parameter &parameter, RNG::SeedType seed = 0);

  VirtualMap(const VirtualMap &other);

  ~VirtualMap() {}

  VirtualMap &operator=(const VirtualMap &other);

  inline Parameter getParameter() const { return parameter_; }

  int getVirtualLandmarkSize() const { return virtual_landmarks_.size(); }

  double explored() const;

  std::vector<VirtualLandmark>::const_iterator cbeginVirtualLandmark() const { return virtual_landmarks_.cbegin(); }

  std::vector<VirtualLandmark>::const_iterator cendVirtualLandnmark() const { return virtual_landmarks_.cend(); }

  std::vector<std::shared_ptr<Map>>::const_iterator cbeginSampledMap() const { return samples_.cbegin(); }

  std::vector<std::shared_ptr<Map>>::const_iterator cendSampledMap() const { return samples_.cend(); }

  inline int rows() const { return rows_; }
  inline int cols() const { return cols_; }
  /// Return an array representing the probabilities
  Eigen::MatrixXd toArray() const;
  Eigen::MatrixXd toCovTrace() const;
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> toCovArray() const;

  int getSampledMapSize() const { return samples_.size(); }

  const Map& getSampledMap(int i) const {
    assert(i >= 0 && i < samples_.size());
    return *samples_.at(i);
  }

  const VirtualLandmark &getVirtualLandmark(int i) const {
    assert(i >= 0 && i < virtual_landmarks_.size());
    return virtual_landmarks_[i];
  }

  /// @deprecated
  std::vector<std::shared_ptr<Map>> sampleMap(const SLAM2D &slam);

  bool predictVirtualLandmark(const VehicleBeliefState &state, VirtualLandmark &virtual_landmark,
                              const BearingRangeSensorModel &sensor_model) const;

  /// Update the probability of virtual landmarks from SLAM2D (rebuild the virtual map)
  void updateProbability(const SLAM2D &slam, const BearingRangeSensorModel &sensor_model);
  void copy_updateProbability(const SLAM2D &slam, const BearingRangeSensorModel &sensor_model);

  /// Update the probability of virtual landmarks from one observation at state (incrementally update the virtual map)
  void updateProbability(const VehicleBeliefState &state, const BearingRangeSensorModel &sensor_model);

  /// @deprecated
  void updateInformation(VirtualLandmark &virtual_landmark, const Map &map,
                         const BearingRangeSensorModel &sensor_model) const;

  /// Update the information of virtual landmarks from Map (rebuild the virtual map)
  void updateInformation(const Map &map, const BearingRangeSensorModel &sensor_model);

  /// Update the information of virtual landmarks from one observation at state (incrementally update the virtual map)
  void updateInformation(const VehicleBeliefState &state, const BearingRangeSensorModel &sensor_model);

  std::vector<int> searchVirtualLandmarkNeighbors(const VehicleBeliefState &state,
                                                  double radius,
                                                  int max_neighbors = -1) const;

  int searchVirtualLandmarkNearest(const VehicleBeliefState &state) const;

 private:
  void initialize();

  /// Estimate the covariance matrix when the correlation between two sources are unknown
  Eigen::Matrix2d covarianceIntersection2D(const Eigen::Matrix2d &m1, const Eigen::Matrix2d &m2) const;

  Parameter parameter_;
  std::shared_ptr<RNG> rng_;
  std::vector<VirtualLandmark> virtual_landmarks_;
  std::shared_ptr<KDTreeR2> virtual_landmarks_kdtree_;

  std::vector<std::shared_ptr<Map>> samples_;

  int rows_;
  int cols_;
  int count_explored_;
};
}
#endif //EM_EXPLORATION_VIRTUALMAP_H
