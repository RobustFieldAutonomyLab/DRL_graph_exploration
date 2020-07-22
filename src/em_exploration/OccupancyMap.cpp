#include "em_exploration/OccupancyMap.h"

namespace em_exploration {

void OccupancyMap::Parameter::print() const {
  std::cout << "Occupancy Map Parameters:" << std::endl;
  std::cout << "  X Range: [" << getMinX() << ", " << getMaxX() << "]" << std::endl;
  std::cout << "  Y Range: [" << getMinY() << ", " << getMaxY() << "]" << std::endl;
  std::cout << "  Safe Distance: " << getSafeDistance() << std::endl;
  std::cout << "  Resolution: " << getResolution() << std::endl;
}

OccupancyMap::OccupancyMap(const Parameter &parameter) : parameter_(parameter) {
  cols_ = static_cast<int>(ceil((parameter_.getMaxX() - parameter_.getMinX()) / parameter_.getResolution()));
  rows_ = static_cast<int>(ceil((parameter_.getMaxY() - parameter_.getMinY()) / parameter_.getResolution()));
  map_ = new double[cols_ * rows_];
  for (int i = 0; i < cols_ * rows_; ++i) {
    map_[i] = LOGODDS_UNKNOWN;
  }
}

OccupancyMap::OccupancyMap(const OccupancyMap &other)
    : parameter_(other.parameter_),
      rows_(other.rows_), cols_(other.cols_),
      map_(new double[cols_ * rows_]) {
  for (int i = 0; i < rows_ * cols_; ++i)
    map_[i] = other.map_[i];
}

OccupancyMap& OccupancyMap::operator=(const OccupancyMap &other) {
  delete[] map_;
  map_ = nullptr;

  parameter_ = other.parameter_;
  rows_ = other.rows_;
  cols_ = other.cols_;
  map_ = new double[rows_ * cols_];
  for (int i = 0; i < rows_ * cols_; ++i)
    map_[i] = other.map_[i];

  return *this;
}

Eigen::MatrixXd OccupancyMap::toArray() const {
  Eigen::MatrixXd array(rows_, cols_);
  for (int row = 0; row < rows_; ++row) {
    for (int col = 0; col < cols_; ++col) {
      array(row, col) = LOGODDS2PROB(map_[row * cols_ + col]);
    }
  }

  return array;
}

void OccupancyMap::update(int row, int col, bool free) {
  if (row >= rows_ || row < 0 || col >= cols_ || col < 0)
    return;
  double logodds0 = map_[row * cols_ + col];
  double logodds = logodds0 + (free ? LOGODDS_FREE : LOGODDS_OCCUPIED);
  logodds = std::min(MAX_LOGODDS, std::max(MIN_LOGODDS, logodds));
  map_[row * cols_ + col] = logodds;
}

void OccupancyMap::update(const VehicleBeliefState &state, const BearingRangeSensorModel &sensor_model) {
  double max_range = sensor_model.getParameter().getMaxRange();
  double resolution = parameter_.getResolution();
  double min_x = parameter_.getMinX();
  double min_y = parameter_.getMinY();

  int origin_row = static_cast<int>(floor((state.pose.y() - min_y) / resolution));
  int origin_col = static_cast<int>(floor((state.pose.x() - min_x) / resolution));

//  int max_offset = static_cast<int>(ceil(max_range / resolution));
//  int min_row = std::min(std::max(0, origin_row - max_offset), rows_ - 1);
//  int max_row = std::min(std::max(0, origin_row + max_offset), rows_ - 1);
//  int min_col = std::min(std::max(0, origin_col - max_offset), cols_ - 1);
//  int max_col = std::min(std::max(0, origin_col + max_offset), cols_ - 1);

  /// Speed up update
  std::vector<int> rows, cols;
  cols.push_back(std::min(std::max(0, origin_col), cols_ - 1));
  rows.push_back(std::min(std::max(0, origin_row), rows_ - 1));
  double min_bearing = sensor_model.getParameter().getMinBearing();
  double max_bearing = sensor_model.getParameter().getMaxBearing();
  double x0 = state.pose.x();
  double y0 = state.pose.y();
  double theta0 = state.pose.theta();
  for (double b = min_bearing; b < max_bearing + 1e-5; b += DEG2RAD(3)) {
    double x = x0 + max_range * cos(theta0 + b);
    double y = y0 + max_range * sin(theta0 + b);
    int row = static_cast<int>(floor((y - min_y) / resolution));
    int col = static_cast<int>(floor((x - min_x) / resolution));

    rows.push_back(std::min(std::max(0, row), rows_ - 1));
    cols.push_back(std::min(std::max(0, col), cols_ - 1));
  }
  int min_row = *std::min_element(rows.cbegin(), rows.cend());
  int max_row = *std::max_element(rows.cbegin(), rows.cend());
  int min_col = *std::min_element(cols.cbegin(), cols.cend());
  int max_col = *std::max_element(cols.cbegin(), cols.cend());

  for (int row = min_row; row <= max_row; ++row) {
    for (int col = min_col; col <= max_col; ++col) {
      /// Speed up update
      if (fabs(map_[row * cols_ + col] - MIN_LOGODDS) < 1e-5)
        continue;

      double x = min_x + resolution * (col + 0.5);
      double y = min_y + resolution * (row + 0.5);
      BearingRangeSensorModel::Measurement m = sensor_model.measure(state.pose, Point2(x, y), false, false);
      if (!sensor_model.checkWithoutMinRange(m))
        continue;

      if (map_[row * cols_ + col] > OCCUPIED_THRESH + 1e-8)
        update(row, col, false);
      else
        update(row, col, true);
    }
  }
}

void OccupancyMap::update(const Map &map, const BearingRangeSensorModel &sensor_model) {
  for (int i = 0; i < cols_ * rows_; ++i) {
    map_[i] = LOGODDS_UNKNOWN;
  }

  for (auto it = map.cbeginLandmark(); it != map.cendLandmark(); ++it) {
    int origin_row = static_cast<int>(floor((it->second.point.y() - parameter_.getMinY()) / parameter_.getResolution()));
    int origin_col = static_cast<int>(floor((it->second.point.x() - parameter_.getMinX()) / parameter_.getResolution()));
    update(origin_row, origin_col, false);
  }

  for (auto it = map.cbeginTrajectory(); it != map.cendTrajectory(); ++it) {
    if (it->core_vehicle) {
      update(*it, sensor_model);
    }
  }
}

}

