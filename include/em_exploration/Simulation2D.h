#ifndef EM_EXPLORATION_SIMULATION2D_H
#define EM_EXPLORATION_SIMULATION2D_H

#include <iostream>
#include <unordered_map>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Rot2.h>

#include "em_exploration/RNG.h"
#include "em_exploration/Distance.h"


namespace em_exploration {

using gtsam::Pose2;
using gtsam::Point2;
using gtsam::Rot2;

/**
 * http://alienryderflex.com/polygon/
 */
class Polygon {
 public:
  Polygon(int num_vertices);

  ~Polygon();

  void setVertex(int i, double x, double y);

  bool pointInPolygon(double x, double y) const;

 private:
  void precalculateValues() const;

  int num_vertices_;
  mutable bool precalculated_;
  double *vx_;
  double *vy_;
  double *constant_;
  double *multiple_;
};

class BearingRangeSensorModel {
 public:

  class Parameter {
   public:
    Parameter() {}

    inline void setBearingNoise(double noise) { bearing_noise_ = Rot2(noise).theta(); }
    inline void setRangeNoise(double noise) { range_noise_ = noise; }
    inline void setMinBearing(double min) { min_bearing_ = Rot2(min).theta(); }
    inline void setMaxBearing(double max) { max_bearing_ = Rot2(max).theta(); }
    inline void setMinRange(double min) { min_range_ = min; }
    inline void setMaxRange(double max) { max_range_ = max; }
    inline double getBearingNoise() const { return bearing_noise_; }
    inline double getRangeNoise() const { return range_noise_; }
    inline double getMinBearing() const { return min_bearing_; }
    inline double getMaxBearing() const { return max_bearing_; }
    inline double getMinRange() const { return min_range_; }
    inline double getMaxRange() const { return max_range_; }

    void print() const;

   private:
    double bearing_noise_;
    double range_noise_;
    double min_bearing_;
    double max_bearing_;
    double min_range_;
    double max_range_;
  };

  class Measurement {
   public:
    typedef Eigen::Matrix<double, 2, 3> HxType;
    typedef Eigen::Matrix<double, 2, 2> HlType;
    typedef Eigen::Matrix<double, 2, 1> SigmasType;

    Measurement() : bearing_(0), range_(0), hasJacobian_(false) {}

    Measurement(double bearing, double range, const SigmasType &sigmas);

    Measurement(double bearing, double range, const SigmasType &sigmas, const HxType &Hx, const HlType &Hl);

    inline bool hasJacobian() const { return hasJacobian_; }
    inline double getBearing() const { return bearing_; }
    inline double getRange() const { return range_; }
    inline SigmasType getSigmas() const { return sigmas_; }
    inline HxType getHx() const { assert(hasJacobian_); return Hx_; }
    inline HlType getHl() const { assert(hasJacobian_); return Hl_; }

    /// Transform the measurement to global frame.
    Point2 transformFrom(const Pose2 &origin) const;

    void print() const;

   private:
    double bearing_;
    double range_;
    SigmasType sigmas_;

    bool hasJacobian_;
    HxType Hx_;
    HlType Hl_;
  };

  BearingRangeSensorModel(const Parameter &parameter) : parameter_(parameter), rng_() {}
  BearingRangeSensorModel(const Parameter &parameter, RNG::SeedType seed) : parameter_(parameter), rng_(seed) {}

  inline Parameter getParameter() const { return parameter_; }

  /// Return true if the measurement is valid given the sensor parameter.
  bool check(const Measurement &m) const;

  /**
   * Return true if the measurement is valid given the sensor parameter.
   * Min range is not used here. This function is for building occupancy maps.
  */
  bool checkWithoutMinRange(const Measurement &m) const;

  /**
   * Given sensor origin and point location, return the measurement.
   * If jacobian is true, the measurement result will have jacobian matrices.
   * If noise is true, Gaussian noise is added to the bearing and range measurements.
   */
  Measurement measure(const Pose2 &pose, const Point2 &point, bool jacobian, bool noise) const;

 private:
  mutable RNG rng_;
  Parameter parameter_;
};

/**
 * A simple control model defined by
 *     x' = x + v * dt * cos(theta) + translation_noise
 *     y' = y + v * dt * sin(theta) + translation_noise
 *     theta' = theta + w * dt + rotation_noise
 */
class SimpleControlModel {
 public:

  class Parameter {
   public:
    Parameter() : translation_noise_(0), rotation_noise_(0) {}

    inline double getTranslationNoise() const { return translation_noise_; }
    inline double getRotationNoise() const { return rotation_noise_; }
    inline void setTranslationNoise(double noise) { translation_noise_ = noise; }
    inline void setRotationNoise(double noise) { rotation_noise_ = Rot2(noise).theta(); }

    void print() const;

   private:
    double translation_noise_;
    double rotation_noise_;
  };

  class ControlState {
   public:
    typedef Eigen::Matrix<double, 3, 3> FxType;
    typedef Eigen::Matrix<double, 3, 1> SigmasType;

    ControlState() : hasJacobian_(false) {}

    ControlState(const Pose2 &pose, const Pose2 &odom, const SigmasType &sigmas);

    ControlState(const Pose2 &pose, const Pose2 &odom, const SigmasType &sigmas, const FxType &Fx1, const FxType &Fx2);

    inline bool hasJacobian() const { return hasJacobian_; }
    inline Pose2 getPose() const { return pose_; }
    inline Pose2 getOdom() const { return odom_; }
    inline SigmasType getSigmas() const { return sigmas_; }
    inline FxType getFx1() const { assert(hasJacobian_); return Fx1_; }
    inline FxType getFx2() const { assert(hasJacobian_); return Fx2_; }

    void print() const;

   private:
    Pose2 pose_; /// the actual pose with added noise
    Pose2 odom_; /// control input
    SigmasType sigmas_;

    bool hasJacobian_;
    FxType Fx1_;
    FxType Fx2_;
  };

  SimpleControlModel(const Parameter &parameter) : parameter_(parameter), rng_() {}
  SimpleControlModel(const Parameter &parameter, RNG::SeedType seed) : parameter_(parameter), rng_(seed) {}

  inline Parameter getParameter() const { return parameter_; }

  /**
   * Given current pose and control input, return the evolved pose.
   * If jacobian is true, the result will have jacobian matrices.
   * If noise is true, Gaussian noise is added to the returned state.
   */
  ControlState evolve(const Pose2 &pose, const Pose2 &odom, bool jacobian, bool noise) const;

 private:
  mutable RNG rng_;
  Parameter parameter_;
};

struct VehicleBeliefState {
  VehicleBeliefState() {}
  VehicleBeliefState(const Pose2 &pose)
      : pose(pose), information(Eigen::Matrix3d::Identity()),
        core_vehicle(true) {}
  VehicleBeliefState(const Pose2 &pose, const Eigen::Matrix3d &I)
      : pose(pose), information(I), core_vehicle(true) {}

  Pose2 pose;
  Eigen::Matrix3d information;
  bool core_vehicle; /// Only core vehicle has uncertainty information.

  Eigen::Matrix3d covariance() const;
  Eigen::Matrix3d globalCovariance() const;

  void print() const;
};

struct LandmarkBeliefState {
  LandmarkBeliefState() {}
  LandmarkBeliefState(const Point2 &point)
      : point(point), information(Eigen::Matrix2d::Identity()) {}
  LandmarkBeliefState(const Point2 &point, const Eigen::Matrix2d &I)
      : point(point), information(I) {}

  Point2 point;
  Eigen::Matrix2d information;

  Eigen::Matrix2d covariance() const;

  void print() const;
};

class Environment {
 public:
  class Parameter {
   public:
    Parameter() {}

    inline double getMinX() const { return min_x_; }
    inline double getMinY() const { return min_y_; }
    inline double getMaxX() const { return max_x_; }
    inline double getMaxY() const { return max_y_; }
    inline double getSafeDistance() const { return safe_distance_; }
    inline void setMinX(double value) { min_x_ = value; }
    inline void setMinY(double value) { min_y_ = value; }
    inline void setMaxX(double value) { max_x_ = value; }
    inline void setMaxY(double value) { max_y_ = value; }
    inline void setSafeDistance(double value) { safe_distance_ = value; }

    void print() const;
    double max_steps;

   private:
    double min_x_;
    double max_x_;
    double min_y_;
    double max_y_;
    double safe_distance_;
  };

  typedef std::unordered_map<unsigned int, LandmarkBeliefState> LandmarkMap;
  typedef std::vector<VehicleBeliefState> Trajectory;

  Environment(const Parameter &parameter);

//  Environment(const Environment &other) = default; // Do not make a copy of the kdtrees.
  Environment(const Environment &other);

//  Environment &operator=(const Environment &other) = default;
  Environment &operator=(const Environment &other);

  ~Environment() {}

  inline Parameter getParameter() const { return parameter_; }

  double getDistance() const;

  void addObstacle(const Polygon &obs);

  void addLandmark(unsigned int i, const Point2 &point);

  void addLandmark(unsigned int i, const LandmarkBeliefState &state);

  void updateLandmark(unsigned int i, const Point2 &point);

  void updateLandmark(unsigned int i, const LandmarkBeliefState &state);

  const LandmarkBeliefState &getLandmark(unsigned int i) const;

  int getLandmarkSize() const { return landmarks_.size(); }

  LandmarkMap::iterator beginLandmark() { return landmarks_.begin(); };

  LandmarkMap::const_iterator cbeginLandmark() const { return landmarks_.cbegin(); };

  LandmarkMap::iterator endLandmark() { return landmarks_.end(); };

  LandmarkMap::const_iterator cendLandmark() const { return landmarks_.cend(); };

  void addVehicle(const Pose2 &pose);

  void addVehicle(const VehicleBeliefState &state);

  void updateVehicle(unsigned int i, const Pose2 &pose);

  void updateVehicle(unsigned int i, const VehicleBeliefState &state);

  const VehicleBeliefState &getVehicle(unsigned int i) const;

  const VehicleBeliefState &getCurrentVehicle() const;

  int getTrajectorySize() const { return trajectory_.size(); };

  Trajectory::iterator beginTrajectory() { return trajectory_.begin(); }

  Trajectory::iterator endTrajectory() { return trajectory_.end(); }

  Trajectory::const_iterator cbeginTrajectory() const { return trajectory_.cbegin(); }

  Trajectory::const_iterator cendTrajectory() const { return trajectory_.cend(); }

  void clear();

  void buildLandmarkKDTree() const;

  void buildTrajectoryKDTree(double angle_weight) const;

  std::vector<unsigned int> searchLandmarkNeighbors(const VehicleBeliefState &state,
                                                    double radius, int max_neighbors = -1) const;

  std::vector<unsigned int> searchTrajectoryNeighbors(const VehicleBeliefState &state, double radius,
                                                      int max_neighbors = -1, double angle_weight = 0.0) const;

  unsigned int searchLandmarkNearest(const VehicleBeliefState &state) const;

  unsigned int searchTrajectoryNearest(const VehicleBeliefState &state, double angle_weight) const;

  bool checkSafety(const VehicleBeliefState &state) const;

 private:
  Parameter parameter_;
  LandmarkMap landmarks_;
  Trajectory trajectory_;

  mutable bool landmark_kdtree_built_;
  mutable KDTreeR2 landmark_kdtree_;

  mutable bool trajectory_kdtree_built_;
  mutable KDTreeSE2 trajectory_kdtree_;
  mutable double angle_weight_;

  mutable std::vector<unsigned int> keys_;

  std::shared_ptr<std::vector<Polygon>> obstacles_;
};

typedef Environment Map;

class Simulator2D {
 public:
  typedef std::vector<std::pair<unsigned int, BearingRangeSensorModel::Measurement>> MeasurementVector;

  Simulator2D(const BearingRangeSensorModel::Parameter &sensor_model_params,
              const SimpleControlModel::Parameter &control_model_params);

  Simulator2D(const BearingRangeSensorModel::Parameter &sensor_model_params,
              const SimpleControlModel::Parameter &control_model_params,
              RNG::SeedType seed);

  void addLandmarks(const std::vector<gtsam::Point2> &landmarks,
                    unsigned int num_random_landmarks,
                    const Environment::Parameter &environment_params);

  void printParameters() const;

  void initializeVehicleState(const Pose2 &pose);

  void initializeVehicleBeliefState(const VehicleBeliefState &state);

  Pose2 getVehicleState() const { return vehicle_.pose; }

  VehicleBeliefState getVehicleBeliefState() const { return vehicle_; }

  static SimpleControlModel::ControlState move(const VehicleBeliefState &state, const Pose2 &odom,
                                               const SimpleControlModel &control_model, bool jacobian, bool noise);

  /// Move the vehicle in the simulation environment given odometry input.
  /// Return whether the command is executed (due to safety check) and the final state.
  /// Set ignore_safety to false if safety check is not necessary (the first return value will always be true).
  std::pair<bool, SimpleControlModel::ControlState> move(const Pose2 &odom, bool ignore_safety = true);

  static MeasurementVector measure(const Environment &environment,
                                   const VehicleBeliefState &vehicle,
                                   const BearingRangeSensorModel &sensor_model,
                                   bool jacobian, bool noise);

  /// Return measurement vector containing the landmark keys and measurements.
  MeasurementVector measure() const;

  /// Return the environment which has access to the actual trajectory and landmark locations.
  const Environment &getEnvironment() const { return environment_; }

  const BearingRangeSensorModel &getSensorModel() const { return sensor_model_; }

  const SimpleControlModel &getControlModel() const { return control_model_; }

 private:
  BearingRangeSensorModel sensor_model_;
  SimpleControlModel control_model_;
  Environment environment_;

  VehicleBeliefState vehicle_;

  mutable RNG rng_;
};

}
#endif //EM_EXPLORATION_SIMULATION2D_H
