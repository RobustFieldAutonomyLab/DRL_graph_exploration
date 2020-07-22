#include "em_exploration/Distance.h"

namespace em_exploration {

double sqDistanceBetweenPoses(const gtsam::Pose2 &pose1, const gtsam::Pose2 &pose2, double angle_weight) {
  double range = pose1.range(pose2);
  double angle = pose1.bearing(pose2).theta();
  return pow(range, 2) + pow(angle * angle_weight, 2);
}

double sqBDistanceBetweenPoses(const gtsam::Pose2 &pose1, const gtsam::Matrix3 &cov1,
                               const gtsam::Pose2 &pose2, const gtsam::Matrix3 &cov2) {
  gtsam::Vector3 e = gtsam::traits<gtsam::Pose2>::Logmap(pose1.between(pose2));
  gtsam::Matrix3 sigma = (cov1 + cov2) / 2.0;
  return 0.125 * e.transpose() * sigma.llt().solve(e)
      + 0.5 * (log(sigma.determinant() + 1e-10) - 0.5 * log(cov1.determinant() + 1e-10) - 0.5 * log(cov2.determinant() + 1e-10));
}

double sqMDistanceBetweenPoses(const gtsam::Pose2 &pose1, const gtsam::Pose2 &pose2, const gtsam::Matrix3 &cov) {
  gtsam::Vector3 e = gtsam::traits<gtsam::Pose2>::Logmap(pose1.between(pose2));
  return e.transpose() * cov.llt().solve(e);
}

int nearestNeighbor(const std::vector<gtsam::Pose2> &poses, const gtsam::Pose2 &pose, double angle_weight) {
  int n = -1;
  double d = std::numeric_limits<double>::max();
  for (int i = 0; i < poses.size(); ++i) {
    double di = sqDistanceBetweenPoses(poses[i], pose, angle_weight);
    if (di < d) {
      d = di;
      n = i;
    }
  }
  return n;
}

std::vector<int> radiusNeighbors(const std::vector<gtsam::Pose2> &poses,
                                 const gtsam::Pose2 &pose,
                                 double radius,
                                 double angle_weight) {
  std::vector<int> n;
  if (radius < 0)
    return n;

  radius *= radius;
  for (int i = 0; i < poses.size(); ++i) {
    double d = sqDistanceBetweenPoses(poses[i], pose, angle_weight);
    if (d < radius)
      n.push_back(i);
  }
  return n;
}

void KDTreeR2::build(const std::vector<gtsam::Point2> &points) {
  data_ = points;
}

void KDTreeR2::addPoints(const std::vector<gtsam::Point2> &points) {
  data_.insert(data_.end(), points.begin(), points.end());
}

int KDTreeR2::queryNearestNeighbor(const gtsam::Point2 &point) const {
  assert(data_.size() > 0);

  int n = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < data_.size(); ++i) {
    double dist = point.distance(data_[i]);
    if (dist < min_dist) {
      min_dist = dist;
      n = i;
    }
  }

  return n;
}

std::vector<int> KDTreeR2::queryRadiusNeighbors(const gtsam::Point2 &point,
                                                double radius,
                                                int max_neighbors) const {
  assert(data_.size() > 0);

  std::vector<int> idx;
  if (radius < 0)
    return idx;

  for (int i = 0; i < data_.size(); ++i) {
    double dist = point.distance(data_[i]);
    if (dist < radius) {
      idx.push_back(i);
      if (max_neighbors > 0 && idx.size() >= max_neighbors)
        break;
    }
  }

  return idx;
}

void KDTreeSE2::build(const std::vector<gtsam::Pose2> &poses, double angle_weight) {
  data_ = poses;
  angle_weight_ = angle_weight;
}

void KDTreeSE2::addPoints(const std::vector<gtsam::Pose2> &poses) {
  data_.insert(data_.end(), poses.begin(), poses.end());
}

int KDTreeSE2::queryNearestNeighbor(const gtsam::Pose2 &pose) const {
  assert(data_.size() > 0);

  int n = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < data_.size(); ++i) {
    double dist = sqDistanceBetweenPoses(pose, data_[i], angle_weight_);
    if (dist < min_dist) {
      min_dist = dist;
      n = i;
    }
  }

  return n;
}

std::vector<int> KDTreeSE2::queryRadiusNeighbors(const gtsam::Pose2 &pose, double radius, int max_neighbors) const {
  assert(data_.size() > 0);

  std::vector<int> idx;
  if (radius < 0)
    return idx;

  for (int i = 0; i < data_.size(); ++i) {
    double dist = sqDistanceBetweenPoses(pose, data_[i], angle_weight_);
    if (sqrt(dist) < radius) {
      idx.push_back(i);
      if (max_neighbors > 0 && idx.size() >= max_neighbors)
        break;
    }
  }

  return idx;
}

}

