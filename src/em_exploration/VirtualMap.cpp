#include "em_exploration/VirtualMap.h"
#include "em_exploration/Utils.h"
#include "em_exploration/OccupancyMap.h"

namespace em_exploration {

VirtualMap::Parameter::Parameter(const Map::Parameter &map_parameter)
    : Map::Parameter(map_parameter), resolution_(2.0), sigma0_(2.0), num_samples_(20) {}

void VirtualMap::Parameter::print() const {
  std::cout << "Virtual Map Parameters:" << std::endl;
  std::cout << "  X Range: [" << getMinX() << ", " << getMaxX() << "]" << std::endl;
  std::cout << "  Y Range: [" << getMinY() << ", " << getMaxY() << "]" << std::endl;
  std::cout << "  Safe Distance: " << getSafeDistance() << std::endl;
  std::cout << "  Resolution: " << getResolution() << std::endl;
  std::cout << "  Sigma0: " << getSigma0() << std::endl;
  std::cout << "  Num of Samples: " << getNumSamples() << std::endl;
}

VirtualMap::VirtualMap(const VirtualMap::Parameter &parameter)
    : parameter_(parameter), rng_(new RNG), rows_(0), cols_(0), virtual_landmarks_kdtree_(new KDTreeR2) {
  initialize();
}

VirtualMap::VirtualMap(const VirtualMap::Parameter &parameter, RNG::SeedType seed)
    : parameter_(parameter), rng_(new RNG(seed)), rows_(0), cols_(0), virtual_landmarks_kdtree_(new KDTreeR2) {
  initialize();
}

VirtualMap::VirtualMap(const VirtualMap &other)
    : parameter_(other.parameter_), rng_(other.rng_), rows_(other.rows_), cols_(other.cols_),
      virtual_landmarks_(other.virtual_landmarks_),
      virtual_landmarks_kdtree_(other.virtual_landmarks_kdtree_), // Do not make a copy of the kdtree
      samples_(other.samples_) {}

VirtualMap &VirtualMap::operator=(const VirtualMap &other) {
  this->parameter_ = other.parameter_;
  this->rng_ = other.rng_;
  this->rows_ = other.rows_;
  this->cols_ = other.cols_;
  this->virtual_landmarks_ = other.virtual_landmarks_;
  this->virtual_landmarks_kdtree_ = other.virtual_landmarks_kdtree_;
  this->samples_ = other.samples_;
  return *this;
}

double VirtualMap::explored() const {
  int count = 0;
  int extg = 20;
  for (int i = 0; i < virtual_landmarks_.size(); ++i) {
    if ((virtual_landmarks_[i].probability < 0.49||virtual_landmarks_[i].probability >0.6) \
        && parameter_.getMinX()+extg <= virtual_landmarks_[i].point.x() \
        && virtual_landmarks_[i].point.x()<= parameter_.getMaxX()-extg \
        && parameter_.getMinY()+extg <= virtual_landmarks_[i].point.y() \
        && virtual_landmarks_[i].point.y()<= parameter_.getMaxY()-extg)
    {count++;}
  }
  return (double) count / count_explored_;
}

void VirtualMap::updateProbability(const SLAM2D &slam, const BearingRangeSensorModel &sensor_model) {
//  std::vector<std::shared_ptr<Map>> maps0 = sampleMap(slam);

  std::vector<std::shared_ptr<Map>> maps;
  for (int n = 0; n < parameter_.getNumSamples(); ++n)
//     maps.push_back(slam.sample().second);
    maps.push_back(std::make_shared<Map>(slam.getMap()));

    for (int i = 0; i < virtual_landmarks_.size(); ++i) {
      virtual_landmarks_[i].probability = 0.0;
    }

    OccupancyMap::Parameter parameter(getParameter());
    parameter.setResolution(parameter_.getResolution());

    OccupancyMap occupancy_map(parameter);
    for (const std::shared_ptr<Map> &map : maps) {
      occupancy_map.update(*map, sensor_model);
      assert(occupancy_map.getMapSize() == virtual_landmarks_.size());

      for (int i = 0; i < virtual_landmarks_.size(); ++i) {
        virtual_landmarks_[i].probability += occupancy_map.getProbability(i) / maps.size();
      }
    }

  /*
    std::vector<int> counts(virtual_landmarks_.size(), 0);
    double radius = sensor_model.getParameter().getMaxRange();
    int num_samples = parameter_.getNumSamples();
    for (const std::shared_ptr<Map> &map : maps) {
      std::vector<bool> map_counts(virtual_landmarks_.size(), false);
      for (int i = 0; i < virtual_landmarks_.size(); ++i) {
        if (map_counts[i])
          continue;
        Point2 point = virtual_landmarks_[i].point;
        std::vector<unsigned int> neigbors
            = map->searchTrajectoryNeighbors(VehicleBeliefState(Pose2(point.x(), point.y(), 0)), radius, -1, 0.0);
        for (unsigned int n : neigbors) {
          Pose2 pose = map->getVehicle(n).pose;
          BearingRangeSensorModel::Measurement m = sensor_model.measure(pose, point, false, true);
          if (sensor_model.checkWithoutMinRange(m))
            map_counts[i] = true;
        }
      }
      for (int i = 0; i < virtual_landmarks_.size(); ++i)
        counts[i] += (int) map_counts[i];
    }

    for (int i = 0; i < virtual_landmarks_.size(); ++i) {
      virtual_landmarks_[i].probability =
          1.0 - (counts[i] + parameter_.getAlpha0()) / (num_samples + parameter_.getAlpha0() + parameter_.getBeta0());
    }
    */
}

void VirtualMap::updateProbability(const VehicleBeliefState &state, const BearingRangeSensorModel &sensor_model) {
  OccupancyMap::Parameter parameter(getParameter());
  parameter.setResolution(parameter_.getResolution());
  OccupancyMap occupancy_map(parameter);
  for (int i = 0; i < virtual_landmarks_.size(); ++i) {
    occupancy_map.setProbability(i, virtual_landmarks_[i].probability);
  }

  occupancy_map.update(state, sensor_model);

  for (int i = 0; i < virtual_landmarks_.size(); ++i) {
    virtual_landmarks_[i].probability = occupancy_map.getProbability(i);
    assert(virtual_landmarks_[i].probability > 0 && virtual_landmarks_[i].probability <= 1);
  }
}

Eigen::MatrixXd VirtualMap::toArray() const {
  Eigen::MatrixXd array(rows_, cols_);
  for (int i = 0; i < virtual_landmarks_.size(); ++i)
    array(i / cols_, i % cols_) = virtual_landmarks_[i].probability;

  return array;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> VirtualMap::toCovArray() const {
  Eigen::MatrixXd array_length(rows_, cols_);
  Eigen::MatrixXd array_angle(rows_, cols_);
  for (int i = 0; i < virtual_landmarks_.size(); ++i) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(virtual_landmarks_[i].covariance());
    double l = sqrt(es.eigenvalues()[1]);
    double a = atan2(es.eigenvectors()(1, 1), es.eigenvectors()(0, 1));
    array_length(i / cols_, i % cols_) = std::min(l, parameter_.getSigma0());
    array_angle(i / cols_, i % cols_) = a;
  }
  return std::make_pair(array_length, array_angle);
}

Eigen::MatrixXd VirtualMap::toCovTrace() const {
    Eigen::MatrixXd array(rows_, cols_);
    for (int i = 0; i < virtual_landmarks_.size(); ++i) {
        array(i / cols_, i % cols_) = virtual_landmarks_[i].covariance().trace();
    }
    return array;
}

std::vector<std::shared_ptr<Map>> VirtualMap::sampleMap(const SLAM2D &slam) {
  samples_.clear();
  const Map &map = slam.getMap();

  bool ignore_non_core_vehicles = true;
  std::vector<unsigned int> pose_keys;
  for (unsigned int i = 0; i < map.getTrajectorySize(); ++i) {
    if (!ignore_non_core_vehicles || map.getVehicle(i).core_vehicle)
      pose_keys.emplace_back(i);
  }
  std::vector<unsigned int> point_keys;
  for (auto it = map.cbeginLandmark(); it != map.cendLandmark(); ++it)
    point_keys.emplace_back(it->first);

  Eigen::MatrixXd cov = slam.jointMarginalCovariance(pose_keys, point_keys);
//  Eigen::MatrixXd cov = info.llt().solve(Eigen::MatrixXd::Identity(info.rows(), info.cols()));
  rng_->setMultivariateNormal(Eigen::VectorXd::Zero(cov.rows()), cov);

  for (int n = 0; n < parameter_.getNumSamples(); ++n) {
    std::shared_ptr<Map> sample(new Map(map.getParameter()));

    Eigen::VectorXd v = rng_->multivariateNormal();
    int row = 0;
    for (auto it = map.cbeginTrajectory(); it != map.cendTrajectory(); ++it) {
      if (ignore_non_core_vehicles) {
        if (it->core_vehicle) {
          VehicleBeliefState state(gtsam::traits<Pose2>::Retract(it->pose, v.segment(row, 3)), it->information);
          state.core_vehicle = true;
          sample->addVehicle(state);
          row += 3;
        } else {
          sample->addVehicle(*it);
        }
      } else {
        VehicleBeliefState state(it->pose.retract(v.segment(row, 3)), it->information);
        state.core_vehicle = it->core_vehicle;
        sample->addVehicle(state);
        row += 3;
      }
    }

    for (auto it = map.cbeginLandmark(); it != map.cendLandmark(); ++it) {
      LandmarkBeliefState state(gtsam::traits<Point2>::Retract(it->second.point, v.segment(row, 2)), it->second.information);
      sample->addLandmark(it->first, state);
      row += 2;
    }

    samples_.push_back(sample);
  }
  return samples_;
}

bool VirtualMap::predictVirtualLandmark(const VehicleBeliefState &state, VirtualLandmark &virtual_landmark,
                                        const BearingRangeSensorModel &sensor_model) const {
  BearingRangeSensorModel::Measurement m = sensor_model.measure(state.pose, virtual_landmark.point, true, false);

  if (!sensor_model.check(m))
    return false;

  Eigen::Matrix2d R = m.getSigmas().asDiagonal();
  R = R * R;
  Eigen::Matrix<double, 2, 3> Hx = m.getHx();
  Eigen::Matrix2d Hl = m.getHl();
  Hl = (Hl.transpose() * Hl).inverse() * Hl.transpose();

  Eigen::Matrix2d cov = Hl * (R + Hx * state.information.llt().solve(Hx.transpose())) * Hl.transpose();
  virtual_landmark.information = inverse(cov);
  return true;
}

/// @deprecated
void VirtualMap::updateInformation(VirtualLandmark &virtual_landmark, const Map &map,
                       const BearingRangeSensorModel &sensor_model) const {
  gtsam::Point2 point = virtual_landmark.point;
  VehicleBeliefState vehicle_state(gtsam::Pose2(point.x(), point.y(), 0.0));
  std::vector<unsigned int> neighbors =
      map.searchTrajectoryNeighbors(vehicle_state, sensor_model.getParameter().getMaxRange(), -1, 0.0);

  bool updated = false;
  for (unsigned int x : neighbors) {
    const VehicleBeliefState &state = map.getVehicle(x);
    VirtualLandmark temp;
    temp.point = virtual_landmark.point;
    if (!predictVirtualLandmark(state, temp, sensor_model))
      continue;

    if (virtual_landmark.updated)
      virtual_landmark.information = covarianceIntersection2D(virtual_landmark.information, temp.information);
    else {
      virtual_landmark.information = temp.information;
      virtual_landmark.updated = true;
    }
  }
}

void VirtualMap::updateInformation(const Map &map, const BearingRangeSensorModel &sensor_model) {
  for (auto &it : virtual_landmarks_) {
    it.updated = false;
    it.information = (Eigen::Matrix2d() << 1.0 / pow(parameter_.getSigma0(), 2), 0,
                      0, 1.0 / pow(parameter_.getSigma0(), 2)).finished();
  }

//    for (VirtualLandmark &l : virtual_landmarks_) {
//      updateInformation(l, map, sensor_model);
//    }
  for (auto it = map.cbeginTrajectory(); it != map.cendTrajectory(); ++it) {
    if (it->core_vehicle) {
      updateInformation(*it, sensor_model);
    }
  }
}

std::vector<int> VirtualMap::searchVirtualLandmarkNeighbors(const VehicleBeliefState &state,
                                                double radius,
                                                int max_neighbors) const {
  return virtual_landmarks_kdtree_->queryRadiusNeighbors(state.pose.t(), radius, max_neighbors);
}

int VirtualMap::searchVirtualLandmarkNearest(const VehicleBeliefState &state) const {
//    return virtual_landmarks_kdtree_->queryNearestNeighbor(state.pose.t());
  double x = state.pose.x();
  double y = state.pose.y();
  int col = static_cast<int>(floor((x - parameter_.getMinX()) / parameter_.getResolution()));
  col = col < 0 ? 0 : (col >= cols_ ? cols_ - 1 : col);
  int row = static_cast<int>(floor((y - parameter_.getMinY()) / parameter_.getResolution()));
  row = row < 0 ? 0 : (row >= rows_ ? rows_ - 1 : row);
  return row * cols_ + col;
}

void VirtualMap::updateInformation(const VehicleBeliefState &state, const BearingRangeSensorModel &sensor_model) {
  assert(virtual_landmarks_kdtree_);

  if (state.information.determinant() < 1e-10)
    return;

  std::vector<int> neighbors = searchVirtualLandmarkNeighbors(state, sensor_model.getParameter().getMaxRange(), -1);

  for (int n : neighbors) {
//    if (virtual_landmarks_[n].probability < 0.49)
//      continue;

    VirtualLandmark temp;
    temp.point = virtual_landmarks_[n].point;
    if (!predictVirtualLandmark(state, temp, sensor_model))
      continue;

    if (virtual_landmarks_[n].updated)
      virtual_landmarks_[n].information =
          covarianceIntersection2D(virtual_landmarks_[n].information, temp.information);
    else {
      virtual_landmarks_[n].information = temp.information;
      virtual_landmarks_[n].updated = true;
    }
  }

}

void VirtualMap::initialize() {
  cols_ = static_cast<int>(floor((parameter_.getMaxX() - parameter_.getMinX())
                                     / parameter_.getResolution()));
  rows_ = static_cast<int>(floor((parameter_.getMaxY() - parameter_.getMinY())
                                     / parameter_.getResolution()));

//  int ext = (int)floor(5.0 / parameter_.getResolution());
  int ext = 0.0;
  int extg = 20;
  std::vector<Point2> points;
  for (int row = ext; row < rows_ - ext; ++row) {
    for (int col = ext; col < cols_ - ext; ++col) {
      double x = (col + 0.5) * parameter_.getResolution() + parameter_.getMinX();
      double y = (row + 0.5) * parameter_.getResolution() + parameter_.getMinY();
      Point2 point(x, y);
      Eigen::Matrix2d information;
      information << 1.0 / pow(parameter_.getSigma0(), 2), 0,
          0, 1.0 / pow(parameter_.getSigma0(), 2);
      virtual_landmarks_.emplace_back(0.5, point, information);

      points.push_back(point);
    }
  }
    count_explored_ = (rows_-extg*2/static_cast<int>(parameter_.getResolution()))*(cols_-extg*2/static_cast<int>(parameter_.getResolution()));

//  count_explored_ = points.size();

//  for (int row = 0; row < rows_; ++row) {
//    for (int col = 0; col < cols_; ++col) {
//      if (row >= ext && row < rows_ - ext && col >= ext && col < cols_ - ext)
//          continue;
//      double x = (col + 0.5) * parameter_.getResolution() + parameter_.getMinX();
//      double y = (row + 0.5) * parameter_.getResolution() + parameter_.getMinY();
//      Point2 point(x, y);
//      Eigen::Matrix2d information;
//      information << 1.0 / pow(parameter_.getSigma0(), 2), 0,
//          0, 1.0 / pow(parameter_.getSigma0(), 2);
//      virtual_landmarks_.emplace_back(0.5, point, information);
//
//      points.push_back(point);
//    }
//  }

  virtual_landmarks_kdtree_->build(points);
}

Eigen::Matrix2d VirtualMap::covarianceIntersection2D(const Eigen::Matrix2d &m1, const Eigen::Matrix2d &m2) const {
  double a = m1.determinant();
  double b = m2.determinant();
  double c = a * m1.llt().solve(m2).trace();
  double d = a + b - c;

  double w = 0.5 * (2 * b - c) / d;
  if ((w < 0 && d < 0) || (w > 1 && d > 0))
    w = 0.0;
  else if ((w < 0 && d > 0) || (w > 1 && d < 0))
    w = 1.0;

  return w * m1 + (1.0 - w) * m2;
}
}
