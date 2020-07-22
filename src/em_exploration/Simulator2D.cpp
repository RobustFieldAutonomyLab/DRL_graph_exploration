#include "em_exploration/Simulation2D.h"
#include "em_exploration/Utils.h"

namespace em_exploration {

using gtsam::Point2;
using gtsam::Pose2;

  Polygon::Polygon(int num_vertices)
      : num_vertices_(num_vertices), precalculated_(false),
        vx_(new double[num_vertices_]()),
        vy_(new double[num_vertices_]()),
        constant_(new double[num_vertices_]()),
        multiple_(new double[num_vertices_]()) {}

  Polygon::~Polygon() {
    delete[] vx_;
    delete[] vy_;
    delete[] constant_;
    delete[] multiple_;
  }

  void Polygon::setVertex(int i, double x, double y) {
    assert(i >= 0 && i < num_vertices_);
    vx_[i] = x;
    vy_[i] = y;
    precalculated_ = false;
  }

  bool Polygon::pointInPolygon(double x, double y) const {
    if (!precalculated_)
      precalculateValues();

    bool odd_nodes = false;

    for (int i = 0, j = num_vertices_ - 1; i < num_vertices_; i++) {
      if ((vy_[i] < y && vy_[j] >= y) || (vy_[j] < y && vy_[i] >= y)) {
        odd_nodes ^= (y * multiple_[i] + constant_[i] < x);
      }
      j = i;
    }

    return odd_nodes;
  }

  void Polygon::precalculateValues() const {
    for (int i = 0, j = num_vertices_ - 1; i < num_vertices_; ++i) {
      if (vy_[j] == vy_[i]) {
        constant_[i] = vx_[i];
        multiple_[i] = 0;
      } else {
        constant_[i] = vx_[i] - (vy_[i] * vx_[j]) / (vy_[j] - vy_[i]) + (vy_[i] * vx_[i]) / (vy_[j] - vy_[i]);
        multiple_[i] = (vx_[j] - vx_[i]) / (vy_[j] - vy_[i]);
      }
      j = i;
    }
    precalculated_ = true;
  }

void BearingRangeSensorModel::Parameter::print() const {
  std::cout << "BearingRangeSensorModel Parameters" << std::endl;
  std::cout << "  Bearing Noise: " << getBearingNoise() << " (rad), " << RAD2DEG(getBearingNoise()) << " (deg)"
            << std::endl;
  std::cout << "  Range Noise: " << getRangeNoise() << " (m)" << std::endl;
  std::cout << "  Min Bearing: " << getMinBearing() << " (rad), " << RAD2DEG(getMinBearing()) << " (deg)"
            << std::endl;
  std::cout << "  Max Bearing: " << getMaxBearing() << " (rad), " << RAD2DEG(getMaxBearing()) << " (deg)"
            << std::endl;
  std::cout << "  Min Range: " << getMinRange() << " (m)" << std::endl;
  std::cout << "  Max Range: " << getMaxRange() << " (m)" << std::endl;
}

typedef BearingRangeSensorModel::Measurement BM;

BM::Measurement(double bearing, double range, const SigmasType &sigmas)
    : bearing_(bearing), range_(range), sigmas_(sigmas), hasJacobian_(false) {}

BM::Measurement(double bearing, double range, const SigmasType &sigmas, const HxType &Hx, const HlType &Hl)
    : bearing_(bearing), range_(range), sigmas_(sigmas),
      hasJacobian_(true), Hx_(Hx), Hl_(Hl) {}

void BM::print() const {
  std::cout << "BearingRangeMeasurements" << std::endl;
  std::cout << "  Bearing: " << bearing_ << " (rad), " << RAD2DEG(bearing_) << " (deg)" << std::endl;
  std::cout << "  Range: " << range_ << " (m)" << std::endl;
  std::cout << "  Sigmas: [" << sigmas_(0) << ", " << sigmas_(1) << "]" << std::endl;
  std::cout << "  Has Jacobian: " << (hasJacobian_ ? "true" : "false") << std::endl;
  if (hasJacobian_) {
    Eigen::IOFormat fmt(4, 0, ", ", "\n", "    [", "]");
    std::cout << "  Jacobian Hx: \n" << Hx_.format(fmt) << std::endl;
    std::cout << "  Jacobian Hl: \n" << Hl_.format(fmt) << std::endl;
  }
}

Point2 BM::transformFrom(const Pose2 &origin) const {
  return origin.transform_from(Point2(range_ * cos(bearing_),
                                      range_ * sin(bearing_)));
}

bool BearingRangeSensorModel::check(const BM &m) const {
  return m.getBearing() < parameter_.getMaxBearing() &&
      m.getBearing() > parameter_.getMinBearing() &&
      m.getRange() < parameter_.getMaxRange() &&
      m.getRange() > parameter_.getMinRange();
}

bool BearingRangeSensorModel::checkWithoutMinRange(const BM &m) const {
  return m.getBearing() < parameter_.getMaxBearing() &&
      m.getBearing() > parameter_.getMinBearing() &&
      m.getRange() < parameter_.getMaxRange();
}

BM BearingRangeSensorModel::measure(const Pose2 &pose, const Point2 &point, bool jacobian, bool noise) const {
  Measurement::SigmasType sigmas;
  sigmas << parameter_.getBearingNoise(), parameter_.getRangeNoise();
  double bearing_noise = noise ? rng_.normal(0.0, sigmas(0)) : 0;
  double range_noise = noise ? rng_.normal(0.0, sigmas(1)) : 0;

  if (!jacobian) {
    double bearing = pose.bearing(point).theta() + bearing_noise;
    double range = pose.range(point) + range_noise;
    return BM(bearing, range, sigmas);
  } else {
    Eigen::MatrixXd Hx_bearing(1, 3), Hl_bearing(1, 2);
    Eigen::MatrixXd Hx_range(1, 3), Hl_range(1, 2);
    double bearing = pose.bearing(point, Hx_bearing, Hl_bearing).theta() + bearing_noise;
    double range = pose.range(point, Hx_range, Hl_range) + range_noise;
    return BM(bearing, range, sigmas,
             (Measurement::HxType() << Hx_bearing, Hx_range).finished(),
             (Measurement::HlType() << Hl_bearing, Hl_range).finished());
  }
}

void SimpleControlModel::Parameter::print() const {
  std::cout << "SimpleControlModel Parameters" << std::endl;
  std::cout << "  Rotation Noise: " << getRotationNoise() << " (rad), " << RAD2DEG(getRotationNoise()) << " (deg)"
            << std::endl;
  std::cout << "  Translation Range: " << getTranslationNoise() << " (m)" << std::endl;
}

typedef SimpleControlModel::ControlState SC;

SC::ControlState(const Pose2 &pose, const Pose2 &odom, const SigmasType &sigmas)
    : pose_(pose), odom_(odom), sigmas_(sigmas), hasJacobian_(false) {}

SC::ControlState(const Pose2 &pose, const Pose2 &odom, const SigmasType &sigmas, const FxType &Fx1, const FxType &Fx2)
    : pose_(pose), odom_(odom), sigmas_(sigmas), hasJacobian_(true), Fx1_(Fx1), Fx2_(Fx2) {}

void SC::print() const {
    std::cout << "ControlState:" << std::endl;
    std::cout << "  Pose: [" << pose_.x() << ", " << pose_.y() << ", " << pose_.theta() << "]" << std::endl;
    std::cout << "  Sigmas: [" << sigmas_(0) << ", " << sigmas_(1) << ", " << sigmas_(2) << "]" << std::endl;
    std::cout << "  Has Jacobian: " << (hasJacobian_ ? "true" : "false") << std::endl;
    if (hasJacobian_) {
      Eigen::IOFormat fmt(4, 0, ", ", "\n", "    [", "]");
      std::cout << "  Jacobian Fx1: \n" << Fx1_.format(fmt) << std::endl;
      std::cout << "  Jacobian Fx2: \n" << Fx2_.format(fmt) << std::endl;
    }
}

SC SimpleControlModel::evolve(const Pose2 &pose, const Pose2 &odom, bool jacobian, bool noise) const {
  ControlState::SigmasType sigmas;
  sigmas << parameter_.getTranslationNoise(),
      parameter_.getTranslationNoise(),
      parameter_.getRotationNoise();

  double x_noise = noise ? rng_.normal(0.0, sigmas(0)) : 0;
  double y_noise = noise ? rng_.normal(0.0, sigmas(1)) : 0;
  double theta_noise = noise ? rng_.normal(0.0, sigmas(2)) : 0;

  Pose2 n(x_noise, y_noise, theta_noise);
  Pose2 new_pose(pose * odom);
  Pose2 new_pose_with_noise(new_pose * n);

  if (jacobian) {
    ControlState::FxType Fx1, Fx2;
    pose.between(new_pose, Fx1, Fx2);
    return ControlState(new_pose_with_noise, odom, sigmas, Fx1, Fx2);
  } else {
    return ControlState(new_pose_with_noise, odom, sigmas);
  }
}

Eigen::Matrix3d VehicleBeliefState::covariance() const {
  return inverse(information);
}

Eigen::Matrix3d VehicleBeliefState::globalCovariance() const {
  Pose2 R(pose.r(), Point2());
  return R.matrix() * inverse(information) * R.matrix().transpose();
}

void VehicleBeliefState::print() const {
  std::cout << "Vehicle Belief State" << std::endl;
  std::cout << "  Pose: [" << pose.x() << ", " << pose.y() << ", " << pose.theta() << "]" << std::endl;
  std::cout << "  Core: " << core_vehicle << std::endl;
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "    [", "]");
  std::cout << "  Information: \n" << covariance().format(fmt) << std::endl;
}

Eigen::Matrix2d LandmarkBeliefState::covariance() const {
  return inverse(information);
}

void LandmarkBeliefState::print() const {
  std::cout << "Landmark Belief State" << std::endl;
  std::cout << "  Point: [" << point.x() << ", " << point.y() << "]" << std::endl;
  Eigen::IOFormat fmt(4, 0, ", ", "\n", "    [", "]");
  std::cout << "  Information: \n" << covariance().format(fmt) << std::endl;
}

void Environment::Parameter::print() const {
  std::cout << "Environment Parameters:" << std::endl;
  std::cout << "  X Range: [" << min_x_ << ", " << max_x_ << "]" << std::endl;
  std::cout << "  Y Range: [" << min_y_ << ", " << max_y_ << "]" << std::endl;
  std::cout << "  Safe Distance: " << safe_distance_ << std::endl;
}

Environment::Environment(const Parameter &parameter)
    : parameter_(parameter),
      landmark_kdtree_built_(false), landmark_kdtree_(),
      trajectory_kdtree_built_(false), angle_weight_(0.0), trajectory_kdtree_(),
      obstacles_(new std::vector<Polygon>()) {}

Environment::Environment(const Environment &other)
    : parameter_(other.parameter_), landmarks_(other.landmarks_), trajectory_(other.trajectory_),
      landmark_kdtree_built_(false), landmark_kdtree_(), keys_(),
      trajectory_kdtree_built_(false), angle_weight_(other.angle_weight_), trajectory_kdtree_(),
      obstacles_(other.obstacles_) {}

Environment& Environment::operator=(const Environment &other) {
  parameter_ = other.parameter_;
  landmarks_ = other.landmarks_;
  trajectory_ = other.trajectory_;
  landmark_kdtree_built_ = false;
  landmark_kdtree_ = KDTreeR2();
  trajectory_kdtree_built_ = false;
  trajectory_kdtree_ = KDTreeSE2();
  keys_.clear();
  obstacles_ = other.obstacles_;
  return *this;
}

double Environment::getDistance() const {
  double distance = 0;
  for (int i = 1; i < getTrajectorySize(); ++i) {
    distance += sqrt(sqDistanceBetweenPoses(getVehicle(i - 1).pose, getVehicle(i).pose, 0.5));
  }
  return distance;
}

void Environment::addObstacle(const Polygon &obs) {
  obstacles_->emplace_back(obs);
}

void Environment::addLandmark(unsigned int i, const Point2 &point) {
  assert(landmarks_.find(i) == landmarks_.end());
  landmarks_.emplace(i, point);
  landmark_kdtree_built_ = false;
}

void Environment::addLandmark(unsigned int i, const LandmarkBeliefState &state) {
  assert(landmarks_.find(i) == landmarks_.end());
  landmarks_.emplace(i, state);
  landmark_kdtree_built_ = false;
}

void Environment::updateLandmark(unsigned int i, const Point2 &point) {
  assert(landmarks_.find(i) != landmarks_.end());
  landmarks_[i].point = point;
  landmark_kdtree_built_ = false;
}

void Environment::updateLandmark(unsigned int i, const LandmarkBeliefState &state) {
  assert(landmarks_.find(i) != landmarks_.end());
  landmarks_[i].point = state.point;
  landmarks_[i].information = state.information;
  landmark_kdtree_built_ = false;
}

const LandmarkBeliefState &Environment::getLandmark(unsigned int i) const {
  auto it = landmarks_.find(i);
  assert(it != landmarks_.end());

  return it->second;
}

void Environment::addVehicle(const Pose2 &pose) {
  trajectory_.emplace_back(pose);
  trajectory_kdtree_built_ = false;
}

void Environment::addVehicle(const VehicleBeliefState &state) {
  trajectory_.emplace_back(state);
  trajectory_kdtree_built_ = false;
}

void Environment::updateVehicle(unsigned int i, const Pose2 &pose) {
  assert(i >= 0 && i < trajectory_.size());
  trajectory_[i].pose = pose;
  trajectory_kdtree_built_ = false;
}

void Environment::updateVehicle(unsigned int i, const VehicleBeliefState &state) {
  assert(i >= 0 && i < trajectory_.size());
  trajectory_[i].pose = state.pose;
  trajectory_[i].information = state.information;
  trajectory_[i].core_vehicle = state.core_vehicle;
  trajectory_kdtree_built_ = false;
}

const VehicleBeliefState &Environment::getVehicle(unsigned int i) const {
  assert(i >= 0 && i < trajectory_.size());
  return trajectory_[i];
}

const VehicleBeliefState &Environment::getCurrentVehicle() const {
  return trajectory_.back();
}

void Environment::clear() {
  landmarks_.clear();
  landmark_kdtree_built_ = false;

  trajectory_.clear();
  trajectory_kdtree_built_ = false;

  keys_.clear();
}

void Environment::buildLandmarkKDTree() const {
  if (landmark_kdtree_built_ || landmarks_.size() == 0)
    return;

  std::vector<Point2> points;
  keys_.clear();
  for (const auto &it : landmarks_) {
    points.emplace_back(it.second.point);
    keys_.emplace_back(it.first);
  }

  landmark_kdtree_.build(points);
  landmark_kdtree_built_ = true;
}

void Environment::buildTrajectoryKDTree(double angle_weight) const {
  if (trajectory_kdtree_built_ || trajectory_.size() == 0)
    return;

  std::vector<Pose2> poses;
  for (const auto &it : trajectory_) {
    poses.emplace_back(it.pose);
  }

  trajectory_kdtree_.build(poses, angle_weight);
  trajectory_kdtree_built_ = true;
  angle_weight_ = angle_weight;
}

std::vector<unsigned int> Environment::searchLandmarkNeighbors(const VehicleBeliefState &state,
                                                  double radius, int max_neighbors) const {
  if (landmarks_.size() == 0)
    return std::vector<unsigned int>();

  if (!landmark_kdtree_built_)
    buildLandmarkKDTree();
    
  std::vector<int> idx = landmark_kdtree_.queryRadiusNeighbors(state.pose.translation(), radius, max_neighbors);

  std::vector<unsigned int> neighbors;
  for (int i : idx) {
    neighbors.emplace_back(keys_[i]);
  }
  return neighbors;
}

std::vector<unsigned int> Environment::searchTrajectoryNeighbors(const VehicleBeliefState &state, double radius,
                                                    int max_neighbors, double angle_weight) const {
  if (trajectory_.size() == 0)
    return std::vector<unsigned int>();

  if (!trajectory_kdtree_built_ || fabs(angle_weight - angle_weight_) > 1e-8)
    buildTrajectoryKDTree(angle_weight);

  std::vector<int> idx = trajectory_kdtree_.queryRadiusNeighbors(state.pose, radius, max_neighbors);

  std::vector<unsigned int> neighbors;
  for (int i : idx) {
    neighbors.emplace_back(i);
  }
  return neighbors;
}

unsigned int Environment::searchLandmarkNearest(const VehicleBeliefState &state) const {
  if (landmarks_.size() == 0)
    return 1;

  if (!landmark_kdtree_built_)
    buildLandmarkKDTree();

  int idx = landmark_kdtree_.queryNearestNeighbor(state.pose.translation());
  return keys_[idx];
}

unsigned int Environment::searchTrajectoryNearest(const VehicleBeliefState &state, double angle_weight) const {
  if (trajectory_.size() == 0)
    return 1;

  if (!trajectory_kdtree_built_ || fabs(angle_weight - angle_weight_) > 1e-8)
    buildTrajectoryKDTree(angle_weight);

  int idx = trajectory_kdtree_.queryNearestNeighbor(state.pose);
  return (unsigned int) idx;
}

bool Environment::checkSafety(const VehicleBeliefState &state) const {
  if (state.pose.x() < parameter_.getMinX() ||
      state.pose.x() > parameter_.getMaxX() ||
      state.pose.y() < parameter_.getMinY() ||
      state.pose.y() > parameter_.getMaxY())
    return false;

  std::vector<int> neighbors = landmark_kdtree_.queryRadiusNeighbors(state.pose.translation(),
                                                                     parameter_.getSafeDistance(), 1);
  return neighbors.size() == 0;
}

Simulator2D::Simulator2D(const BearingRangeSensorModel::Parameter &sensor_model_params,
            const SimpleControlModel::Parameter &control_model_params)
    : vehicle_(Pose2(0, 0, 0)),
      sensor_model_(sensor_model_params),
      control_model_(control_model_params),
      environment_(Environment::Parameter()),
      rng_() {}

Simulator2D::Simulator2D(const BearingRangeSensorModel::Parameter &sensor_model_params,
            const SimpleControlModel::Parameter &control_model_params,
            RNG::SeedType seed)
    : vehicle_(Pose2(0, 0, 0)),
      sensor_model_(sensor_model_params, seed),
      control_model_(control_model_params, seed),
      environment_(Environment::Parameter()),
      rng_(seed) {}

void Simulator2D::addLandmarks(const std::vector<gtsam::Point2> &landmarks,
                               unsigned int num_random_landmarks,
                               const Environment::Parameter &environment_params){
  environment_ = Environment(environment_params);
  for (int i = 0; i < landmarks.size(); ++i) {
    environment_.addLandmark(i, landmarks[i]);
  }
  for (unsigned int i = 0; i < num_random_landmarks;) {
    double x = rng_.uniformReal(environment_params.getMinX(), environment_params.getMaxX());
    double y = rng_.uniformReal(environment_params.getMinY(), environment_params.getMaxY());

    Point2 point(x, y);
    // Don't sample points near initial location.

    if (point.distance(vehicle_.pose.t()) < 2.0)
      continue;
    environment_.addLandmark(i + landmarks.size(), point);
    i++;
  }
}

void Simulator2D::printParameters() const {
  std::cout << "----------------------------" << std::endl;
  sensor_model_.getParameter().print();
  std::cout << "----------------------------" << std::endl;
  control_model_.getParameter().print();
  std::cout << "----------------------------" << std::endl;
  environment_.getParameter().print();
  std::cout << "----------------------------" << std::endl;
}

void Simulator2D::initializeVehicleState(const Pose2 &pose) {
  vehicle_.pose = pose;
  environment_.addVehicle(pose);
}

void Simulator2D::initializeVehicleBeliefState(const VehicleBeliefState &state) {
  vehicle_ = state;
  environment_.addVehicle(state);
}

SC Simulator2D::move(const VehicleBeliefState &state, const Pose2 &odom,
                                             const SimpleControlModel &control_model, bool jacobian, bool noise) {
  return control_model.evolve(state.pose, odom, jacobian, noise);
}

std::pair<bool, SC> Simulator2D::move(const Pose2 &odom, bool ignore_safety) {
  SC s = Simulator2D::move(vehicle_, odom, control_model_, true, true);

  VehicleBeliefState v(s.getPose());
  if (!ignore_safety && !environment_.checkSafety(v)) {
    std::cerr << "Control input is invalid!" << std::endl;
    return std::make_pair(false, s);
  }

  vehicle_ = v;
  environment_.addVehicle(vehicle_);
  return std::make_pair(true, s);
}

Simulator2D::MeasurementVector Simulator2D::measure(const Environment &environment,
                                 const VehicleBeliefState &vehicle,
                                 const BearingRangeSensorModel &sensor_model,
                                 bool jacobian, bool noise) {
  MeasurementVector ms;
  std::vector<unsigned int> neighbors =
      environment.searchLandmarkNeighbors(vehicle, sensor_model.getParameter().getMaxRange());

//    for (auto l = environment_.cbeginLandmark(); l != environment_.cendLandmark(); ++l) {
  for (unsigned int key : neighbors) {
    const LandmarkBeliefState &landmark = environment.getLandmark(key);

    BearingRangeSensorModel::Measurement m =
        sensor_model.measure(vehicle.pose, landmark.point, jacobian, noise);
    if (sensor_model.check(m))
      ms.emplace_back(key, m);
  }
  return ms;
}

Simulator2D::MeasurementVector Simulator2D::measure() const {
  return Simulator2D::measure(environment_, vehicle_, sensor_model_, false, true);
}

}

