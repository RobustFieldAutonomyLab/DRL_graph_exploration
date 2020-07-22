#include "em_exploration/Planner2D.h"
#include <ctime>
#include <pyport.h>

#define LEAFONLY
#define USE_FAST_MARGINAL2

namespace em_exploration {

void EMPlanner2D::Parameter::print() const {
  std::cout << "RRT Planner Parameters:" << std::endl;
  std::cout << "  Verbose: " << verbose << std::endl;
  std::cout << "  Seed: " << seed << std::endl;
  std::cout << "  Angle Weight: " << angle_weight << std::endl;
  std::cout << "  Distance Weight: [" << distance_weight0 << ", " << distance_weight1 << "]" << std::endl;
  std::cout << "  Max Nodes: " << max_nodes << std::endl;
  std::cout << "  Max Edge Length: " << max_edge_length << std::endl;
  std::cout << "  Occupancy Threshold: " << occupancy_threshold << std::endl;
  std::cout << "  Algorithm: " << algorithm << std::endl;
  std::cout << "  Safe Distance: " << safe_distance << std::endl;
  std::cout << "  Alpha: " << alpha << std::endl;
  std::cout << "  Dubins Control Model Enabled: " << dubins_control_model_enabled << std::endl;
  std::cout << "    Max Angular Velocity: " << dubins_parameter.max_w << ", Delta: "
			<< dubins_parameter.dw << std::endl;
  std::cout << "    Translational Velocity: [" << dubins_parameter.min_v << ", " << dubins_parameter.max_v
			<< "], Delta: " << dubins_parameter.dv
			<< std::endl;
  std::cout << "    Duration: [" << dubins_parameter.min_duration << ", " << dubins_parameter.max_duration
			<< "], Delta: "
			<< dubins_parameter.dt
			<< std::endl;
  std::cout << "    Tolerance Radius: " << dubins_parameter.tolerance_radius << std::endl;
}

EMPlanner2D::EMPlanner2D(const Parameter &parameter,
						 const BearingRangeSensorModel &sensor_model,
						 const SimpleControlModel &control_model)
	: parameter_(parameter), sensor_model_(sensor_model), control_model_(control_model),
	  rng_(parameter_.seed), qrng_(3),
	  update_distance_weight_(true), distance_weight_(1e10) {
  if (parameter_.dubins_control_model_enabled) {
	qrng_.setDim(2);
	initializeDubinsPathLibrary();
  }
  qrng_.setCount(rng_.uniformInt(0, 100000));
}

bool EMPlanner2D::isSafe(EMPlanner2D::Node::shared_ptr node) const {
  // int vl_nearest_neighbor = virtual_map_->searchVirtualLandmarkNearest(node->state);
  // if (vl_nearest_neighbor < virtual_map_->getVirtualLandmarkSize() &&
  //     virtual_map_->getVirtualLandmark(vl_nearest_neighbor).probability > parameter_.occupancy_threshold)
  //   return false;
  std::vector<unsigned int> l_neighbors =
	  map_->searchLandmarkNeighbors(node->state, parameter_.safe_distance, 1);
  return l_neighbors.size() == 0;
}

bool EMPlanner2D::isSafe(EMPlanner2D::Node::shared_ptr node, EMPlanner2D::Node::shared_ptr parent) const {
  double safe_distance = parameter_.safe_distance;
  if (fabs(safe_distance) < 1e-3)
	return true;
  // assert(safe_distance > 0);

  if (parameter_.dubins_control_model_enabled) {
	for (int i = 1; i < node->poses.size() - 1; ++i) {
	  EMPlanner2D::Node::shared_ptr waypoint(new EMPlanner2D::Node(node->poses[i]));
	  if (!isSafe(waypoint))
		return false;
	}
	return true;
  } else {
	const Pose2 &pose1 = node->state.pose;
	const Pose2 &pose2 = parent->state.pose;
	double d = pose1.range(pose2);
	Point2 unit = 1.0 / d * (pose2.transform_to(pose1.t()));

	for (double l = safe_distance / 2; l < d; l += safe_distance / 2) {
	  Point2 point_between = pose1.transform_from(l * unit);
	  Pose2 pose_between(pose1.r(), point_between);
	  EMPlanner2D::Node::shared_ptr temp(new EMPlanner2D::Node(pose_between));
	  if (!isSafe(temp))
		return false;
	}
	return true;
  }
}

bool EMPlanner2D::isReached(EMPlanner2D::Node::shared_ptr node, EMPlanner2D::Node::shared_ptr goal) const{
  bool reached = false;
  double node_x = node ->state.pose.x();
  double node_y = node ->state.pose.y();
  double goal_x = goal ->state.pose.x();
  double goal_y = goal ->state.pose.y();
  double dist = sqrt(pow((node_x - goal_x), 2) + pow((node_y-goal_y),2));
  if (dist<=parameter_.max_edge_length){
	reached = true;
  }
  return reached;
}

EMPlanner2D::Node::shared_ptr EMPlanner2D::sampleNode() const {
  const Map::Parameter &map_parameter = map_->getParameter();

  int failed = 0;
  while (true) {
	Eigen::VectorXd value = qrng_.generate();
	double x = map_parameter.getMinX() + value(0) * (map_parameter.getMaxX() - map_parameter.getMinX());
	double y = map_parameter.getMinY() + value(1) * (map_parameter.getMaxY() - map_parameter.getMinY());

	double theta = 0;
	if (!parameter_.dubins_control_model_enabled) {
	  theta = value(2) * 2.0 * M_PI;
	}

	Pose2 pose(x, y, theta);
	EMPlanner2D::Node::shared_ptr node(new EMPlanner2D::Node(pose));
	if (isSafe(node))
	  return node;
	else
	  failed += 1;

	if (failed > 1000)
	  return nullptr;
  }
}

bool EMPlanner2D::connectNodeDubinsPath(EMPlanner2D::Node::shared_ptr node, EMPlanner2D::Node::shared_ptr parent) {
  if (parameter_.verbose)
	std::cout << "Connect Dubins Path." << std::endl;

  /*
  const double x1 = 0;
  const double y1 = 0;
  Point2 local = parent->state.pose.transform_to(node->state.pose.t());
  double x2 = local.x();
  double y2 = local.y();
  double dx12 = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));

  for (const auto &dubins : dubins_library_) {
	double x = dubins.back().x();
	double y = dubins.back().y();
	double dist_to_line = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / dx12;

	if (dist_to_line < parameter_.dubins_parameter.tolerance_radius) {
	  node->dubins.clear();
	  node->dubins.emplace_back(parent->state.pose);
	  for (int i = 1; i < dpath.poses.size(); ++i) {
		node->dubins.emplace_back(node->dpath.poses.back() * dpath.poses[i - 1].between(dpath.poses[i]));
	  }
	  node->state.pose = node->dubins.back();
	  return true;
	}
  }
  return false;
   */

  for (int i = 0; i < dubins_library_.size(); ++i) {
	Point2 local = parent->state.pose.transform_to(node->state.pose.t());
	if (local.distance(dubins_library_[i].end.t()) < parameter_.dubins_parameter.tolerance_radius) {
	  const Dubins &d = dubins_library_[i];
	  Pose2 pose = parent->state.pose;
	  double dt = parameter_.dubins_parameter.dt;
	  node->poses.clear();
	  for (int step = 0; step < d.num_steps; ++step) {
		double x = pose.x() + d.v * dt * pose.r().c();
		double y = pose.y() + d.v * dt * pose.r().s();
		double theta = pose.theta() + d.w * dt;
		pose = Pose2(x, y, theta);
		node->poses.emplace_back(pose);
	  }
	  node->state.pose = pose;
	  node->n_dubins = i;
	  return true;
	}
  }
  return false;
}

bool EMPlanner2D::connectNode(EMPlanner2D::Node::shared_ptr node, EMPlanner2D::Node::shared_ptr parent) {
  if (parameter_.verbose) {
	std::cout << "Try to connect nodes." << std::endl;
	std::cout << "  Parent: ";
	parent->print();
	std::cout << "  Node: ";
	node->print();
  }

  const Pose2 &origin = parent->state.pose;
  double max_edge_distance = parameter_.max_edge_length;
  double d = origin.range(node->state.pose);

////  following is for constant length step

//  if (fabs(d - parameter_.max_edge_length) > 0 * parameter_.max_edge_length)
//    return false;

  if (!parameter_.dubins_control_model_enabled) {
	Point2 local = origin.transform_to(node->state.pose.t());
	double angle = Rot2::relativeBearing(local).theta();
	//control moving direction
	if (fabs(angle) > M_PI)
	  return false;
//    if (parameter_.reg_out)
//        angle = round(angle / (2 * M_PI / parameter_.num_actions)) * 2 * M_PI / parameter_.num_actions;
	if (d > max_edge_distance){
	local = Point2(parameter_.max_edge_length * cos(angle), parameter_.max_edge_length * sin(angle));
	}
	  else{
	local = Point2(d * cos(angle), d * sin(angle));
	}
	node->state.pose = origin * Pose2(Rot2(angle), local);
	if (parameter_.verbose) {
	  std::cout << "  Scale to ";
	  node->print();
	}
  }

//   if (d > max_edge_distance) {
//     Point2 unit = 1.0 / d * (node->state.pose.t() - origin.t());
//     Point2 point_between = origin.t() + max_edge_distance * unit;
//     node->state.pose = Pose2(node->state.pose.r(), point_between);
//     if (parameter_.verbose) {
//       std::cout << "  Scale to ";
//       node->print();
//     }
//   }

  bool dubins_connected = true;
  if (parameter_.dubins_control_model_enabled)
	dubins_connected = connectNodeDubinsPath(node, parent);
  else
	node->poses.push_back(node->state.pose);

  if (parameter_.verbose) {
	std::cout << "  Done. Connected: " << dubins_connected << std::endl;
	std::cout << "  Node: ";
	node->print();
  }

  if (!dubins_connected)
	return false;

  bool safe = isSafe(node, parent);
  if (parameter_.verbose)
	std::cout << "Safety Checked: " << safe << std::endl;

  if (!safe)
	return false;

  node->parent = parent;
  if (parameter_.dubins_control_model_enabled)
	node->key = static_cast<unsigned int>(parent->key + node->poses.size());
  else
	node->key = parent->key + 1;

//  node->map.reset(new Map(*parent->map));
//  node->virtual_map.reset(new VirtualMap(*parent->virtual_map));
  node->distance = parent->distance + distanceBetweenNodes(node, parent);

  if (parameter_.verbose) {
	std::cout << "  Node connected: ";
	node->print();
  }
  return true;
}

EMPlanner2D::Node::shared_ptr EMPlanner2D::nearestNode(EMPlanner2D::Node::shared_ptr node) const {
  std::vector<Pose2> poses;
  for (auto node: nodes_)
	poses.push_back(node->state.pose);

  int n = nearestNeighbor(poses, node->state.pose, parameter_.angle_weight);
  assert(n >= 0);

  return nodes_[n];
}

std::vector<EMPlanner2D::Node::shared_ptr> EMPlanner2D::neighborNodes(EMPlanner2D::Node::shared_ptr node,
																	  double radius) const {
  std::vector<Pose2> poses;
  for (auto no : nodes_) {
	if (no != node)
	  poses.push_back(no->state.pose);
  }
  std::vector<int> keys = radiusNeighbors(poses, node->state.pose, radius, parameter_.angle_weight);

  std::vector<EMPlanner2D::Node::shared_ptr> nodes;
  for (int k : keys)
	nodes.push_back(nodes_[k]);
  return nodes;
}

double EMPlanner2D::distanceBetweenNodes(EMPlanner2D::Node::shared_ptr node1,
										 EMPlanner2D::Node::shared_ptr node2) const {
  if (parameter_.dubins_control_model_enabled) {
	double v = dubins_library_[node1->n_dubins].v;
	double w = dubins_library_[node1->n_dubins].w;
	double num_steps = dubins_library_[node1->n_dubins].num_steps;
	double dt = parameter_.dubins_parameter.dt;
	return v * dt * num_steps + fabs(w * dt * num_steps) * parameter_.angle_weight;
  } else
	return sqrt(sqDistanceBetweenPoses(node1->state.pose, node2->state.pose, parameter_.angle_weight));
}

double EMPlanner2D::calculateUncertainty(EMPlanner2D::Node::shared_ptr node) const {
  switch (parameter_.algorithm) {
	case OptimizationAlgorithm::EM_DOPT:
	  return calculateUncertainty_EM(node);
	case OptimizationAlgorithm::EM_AOPT:
	  return calculateUncertainty_EM(node);
	case OptimizationAlgorithm::OG_SHANNON:
	  return calculateUncertainty_OG_SHANNON(node);
	case OptimizationAlgorithm::SLAM_OG_SHANNON:
	  return calculateUncertainty_SLAM_OG_SHANNON(node);
	default:
	  std::cout << "Unknown algorithm " << parameter_.algorithm << std::endl;
	  exit(-1);
  }
}

double EMPlanner2D::calculateUncertainty_EM(Node::shared_ptr node) const {
  if (parameter_.verbose)
	std::cout << "Calculate uncertainty - EM." << std::endl;
  assert(node->virtual_map != nullptr);

  double uncertainty = 0.0;
  for (auto it = node->virtual_map->cbeginVirtualLandmark();
	   it != node->virtual_map->cendVirtualLandnmark(); ++it) {
//    double weight = it->probability;
	double weight = it->probability > 0.49 ? 1.0 : 0.0;
	if (parameter_.algorithm == OptimizationAlgorithm::EM_DOPT)
	  uncertainty += weight / it->information.determinant();
	else
	  uncertainty += weight * it->covariance().trace();
  }

  if (parameter_.verbose)
	std::cout << "  Done. Uncertainty: " << uncertainty << std::endl;

  return uncertainty;
}

double EMPlanner2D::calculateUncertainty(const VirtualMap &virtual_map) {
  double uncertainty = 0.0;
  for (auto it = virtual_map.cbeginVirtualLandmark();
	   it != virtual_map.cendVirtualLandnmark(); ++it) {
//	 if (it->probability > 0.49) {
	  uncertainty += 1.0 * it->covariance().trace();
//	 }
  }
  return uncertainty;
}

double EMPlanner2D::calculateUtility(const VirtualMap &virtual_map, double distance, const Parameter &parameter) {
  int vl_known = 0;
  for (auto it = virtual_map.cbeginVirtualLandmark(); it != virtual_map.cendVirtualLandnmark(); ++it)
	if (it->probability < parameter.occupancy_threshold)
	  vl_known++;

  double percentage_known = (double) vl_known / virtual_map.getVirtualLandmarkSize();

  double distance_weight = parameter.distance_weight0
		- (parameter.distance_weight0 - parameter.distance_weight1) * percentage_known;

  return EMPlanner2D::calculateUncertainty(virtual_map) + distance * distance_weight;
}

double EMPlanner2D::calculateUncertainty_OG_SHANNON(Node::shared_ptr node) const {
  double uncertainty = 0.0;

//   double trajectory_entropy = 0.0;
//   double c = 3 * log(2 * M_PI * M_E);
//    for (auto it = node->map->cbeginTrajectory(); it != node->map->cendTrajectory(); ++it) {
//      double det = it->covariance().determinant();
//      trajectory_entropy += 0.5 * (c + log(det));
//    }
//    uncertainty += trajectory_entropy / node->map->getTrajectorySize();

  SLAM2D slam(map_->getParameter());
  slam.fromISAM2(node->isam, *node->map, node->isam->getLinearizationPoint());
  node->virtual_map->updateProbability(slam, sensor_model_);

  double og_entropy = 0.0;
  for (auto it = node->virtual_map->cbeginVirtualLandmark();
	   it != node->virtual_map->cendVirtualLandnmark(); ++it) {
	double p = it->probability;
	og_entropy += -p * log(p) - (1 - p) * log(1 - p);
  }
  uncertainty += og_entropy;

  return uncertainty;
}

double EMPlanner2D::calculateUncertainty_SLAM_OG_SHANNON(Node::shared_ptr node) const {
  double uncertainty = 0.0;

  SLAM2D slam(map_->getParameter());
  slam.fromISAM2(node->isam, *node->map, node->isam->getLinearizationPoint());
  node->virtual_map->updateProbability(slam, sensor_model_);

  double og_entropy = 0.0;
  for (auto it = node->virtual_map->cbeginVirtualLandmark();
	   it != node->virtual_map->cendVirtualLandnmark(); ++it) {
	double p = it->probability;
	og_entropy += -p * log(p) - (1 - p) * log(1 - p);
  }
  uncertainty += w2_ * og_entropy;

  double slam_uncertainty = 0.0;
  for (auto it = node->map->cbeginLandmark(); it != node->map->cendLandmark(); ++it) {
	slam_uncertainty += sqrt(it->second.covariance().determinant());
  }
  uncertainty += w1_ * slam_uncertainty;

  return uncertainty;
}

double EMPlanner2D::costFunction(Node::shared_ptr node) const {
  return node->uncertainty + node->distance * distance_weight_;
}

/*
/// @deprecated
void EMPlanner2D::updateTrajectory(EMPlanner2D::Node::shared_ptr leaf) {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial_estimate;

  EMPlanner2D::Node::shared_ptr node = leaf;

  while (node != nullptr) {
	assert(node->odometry_factors.size() == node->poses.size());

	for (int i = 0; i < node->odometry_factors.size(); ++i) {
	  auto factor = node->odometry_factors[i];
	  graph.add(factor);

	  initial_estimate.insert(factor->back(), node->poses[i]);
	}

	for (auto factor : node->measurement_factors) {
	  graph.add(factor);
	}
	node = node->parent.lock();
  }

  initial_estimate.insert(*values_);
  gtsam::ISAM2Params params;
  params.enableRelinearization = false;
  gtsam::ISAM2 isam(params);
  isam.update(graph, initial_estimate);

  for (auto it = leaf->map->beginLandmark(); it != leaf->map->endLandmark(); ++it) {
	Eigen::Matrix2d cov = isam.marginalCovariance(SLAM2D::getLandmarkSymbol(it->first));
	it->second.information = inverse(cov);
  }

  for (auto it = leaf->map->beginTrajectory(); it != leaf->map->endTrajectory(); ++it) {
	if (!it->core_vehicle)
	  continue;
	unsigned int x = static_cast<unsigned int>(std::distance(leaf->map->beginTrajectory(), it));
	Eigen::Matrix3d cov = isam.marginalCovariance(SLAM2D::getVehicleSymbol(x));
	Pose2 R(it->pose.r(), Point2());
	cov = R.matrix() * cov * R.matrix().transpose();
	it->information = inverse(cov);
  }

  leaf->state.information = leaf->map->getCurrentVehicle().information;
}
 */


void EMPlanner2D::updateTrajectory_EM(Node::shared_ptr leaf) {
  if (parameter_.verbose)
	std::cout << "Optimize trajectory." << std::endl;

  gtsam::NonlinearFactorGraph odom_graph;
  gtsam::NonlinearFactorGraph meas_graph;
  gtsam::Values initial_estimate;
  gtsam::KeySet updated_keys = updated_keys_;

  std::stack<Node::shared_ptr> s;
  EMPlanner2D::Node::shared_ptr node = leaf;
  while (node != root_) {
	if (node->isam != nullptr)
	  break;
	s.push(node);
	node = node->parent.lock();
  }
  while (!s.empty()) {
	node = s.top();
	s.pop();

	for (int i = 0; i < node->odometry_factors.size(); ++i) {
	  auto factor = node->odometry_factors[i];
	  odom_graph.add(factor);
	  initial_estimate.insert(factor->back(), node->poses[i]);
	}
	updated_keys.insert(node->odometry_factors.back()->back());
	meas_graph.add(node->measurement_factors);
  }

#ifdef USE_FAST_MARGINAL2
  FastMarginals2 fm2(slam_->getMarginals());
  fm2.update(odom_graph, meas_graph, initial_estimate, updated_keys);
#else
  gtsam::NonlinearFactorGraph graph;
  graph.add(odom_graph);
  graph.add(meas_graph);
  gtsam::ISAM2 isam2(*root_->isam);
  isam2.update(graph, initial_estimate);
#endif

  leaf->map.reset(new Map(root_->map->getParameter()));
  for (gtsam::Key key : updated_keys) {
#ifdef USE_FAST_MARGINAL2
	Eigen::Matrix3d cov = fm2.marginalCovariance(key);
#else
	Eigen::Matrix3d cov = isam2.marginalCovariance(key);
#endif
	if (initial_estimate.exists(key))
	  leaf->map->addVehicle(VehicleBeliefState(initial_estimate.at<Pose2>(key), cov.inverse()));
	else
	  leaf->map->addVehicle(VehicleBeliefState(values_->at<Pose2>(key), cov.inverse()));
  }
#ifdef USE_FAST_MARGINAL2
  leaf->state.information = fm2.marginalCovariance(SLAM2D::getVehicleSymbol(leaf->key).key()).inverse();
#else
  leaf->state.information = isam2.marginalCovariance(SLAM2D::getVehicleSymbol(leaf->key).key()).inverse();
#endif

//  leaf->isam.reset(new gtsam::ISAM2(*node->isam));
//  leaf->isam->update(graph, initial_estimate);
//  const gtsam::ISAM2 &isam = *leaf->isam;

//  for (auto it = leaf->map->beginLandmark(); it != leaf->map->endLandmark(); ++it) {
//    Eigen::Matrix2d cov = isam.marginalCovariance(SLAM2D::getLandmarkSymbol(it->first));
//    it->second.information = inverse(cov);
//  }
//
//  for (auto it = leaf->map->beginTrajectory(); it != leaf->map->endTrajectory(); ++it) {
//    if (!it->core_vehicle)
//      continue;
//    unsigned int x = static_cast<unsigned int>(std::distance(leaf->map->beginTrajectory(), it));
//    Eigen::Matrix3d cov = isam.marginalCovariance(SLAM2D::getVehicleSymbol(x));
//    Pose2 R(it->pose.r(), Point2());
//    cov = R.matrix() * cov * R.matrix().transpose();
//    it->information = inverse(cov);
//  }

//  leaf->state.information = leaf->map->getCurrentVehicle().information;
}

void EMPlanner2D::updateTrajectory_OG_SHANNON(Node::shared_ptr leaf) {
  if (parameter_.verbose)
	std::cout << "Optimize trajectory." << std::endl;

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial_estimate;

  EMPlanner2D::Node::shared_ptr node = leaf;
  for (int i = 0; i < node->odometry_factors.size(); ++i) {
	auto factor = node->odometry_factors[i];
	graph.add(factor);

	initial_estimate.insert(factor->back(), node->poses[i]);
  }

  for (auto factor : node->measurement_factors)
	graph.add(factor);

  node = node->parent.lock();
  assert(node->isam);

  leaf->isam.reset(new gtsam::ISAM2(*node->isam));
  leaf->isam->update(graph, initial_estimate);

  unsigned int x = static_cast<unsigned int>(leaf->map->getTrajectorySize() - 1);
  Eigen::Matrix3d cov = leaf->isam->marginalCovariance(SLAM2D::getVehicleSymbol(x));
  Pose2 R(leaf->state.pose.r(), Point2());
  cov = R.matrix() * cov * R.matrix().transpose();
  leaf->state.information = inverse(cov);
}

void EMPlanner2D::updateTrajectory_SLAM_OG_SHANNON(Node::shared_ptr leaf) {
  if (parameter_.verbose)
	std::cout << "Optimize trajectory SLAM OG SHANNON." << std::endl;

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial_estimate;

  EMPlanner2D::Node::shared_ptr node = leaf;
  for (int i = 0; i < node->odometry_factors.size(); ++i) {
	auto factor = node->odometry_factors[i];
	graph.add(factor);

	initial_estimate.insert(factor->back(), node->poses[i]);
  }

  for (auto factor : node->measurement_factors)
	graph.add(factor);

  node = node->parent.lock();
  assert(node->isam);

  leaf->isam.reset(new gtsam::ISAM2(*node->isam));
  leaf->isam->update(graph, initial_estimate);

  for (auto it = leaf->map->beginLandmark(); it != leaf->map->endLandmark(); ++it) {
	Eigen::Matrix2d cov = leaf->isam->marginalCovariance(SLAM2D::getLandmarkSymbol(it->first));
	it->second.information = inverse(cov);
  }

  unsigned int x = static_cast<unsigned int>(leaf->map->getTrajectorySize() - 1);
  Eigen::Matrix3d cov = leaf->isam->marginalCovariance(SLAM2D::getVehicleSymbol(x));
  Pose2 R(leaf->state.pose.r(), Point2());
  cov = R.matrix() * cov * R.matrix().transpose();
  leaf->state.information = inverse(cov);
}

void EMPlanner2D::updateNodeInformation(Node::shared_ptr node) {
  switch (parameter_.algorithm) {
	case OptimizationAlgorithm::EM_DOPT:
	  updateNodeInformation_EM(node);
	  break;
	case OptimizationAlgorithm::EM_AOPT:
	  updateNodeInformation_EM(node);
	  break;
	case OptimizationAlgorithm::OG_SHANNON:
	  updateNodeInformation_OG_SHANNON(node);
	  break;
	case OptimizationAlgorithm::SLAM_OG_SHANNON:
	  updateNodeInformation_OG_SHANNON(node);
	  break;
	default:
	  std::cout << "Unknown algorithm " << parameter_.algorithm << std::endl;
	  exit(-1);
  }
}

void EMPlanner2D::updateNodeOccupancyMap(Node::shared_ptr node) {
  if (parameter_.verbose)
	std::cout << "Update node's occupancy map." << std::endl;

  if (parameter_.dubins_control_model_enabled) {
	node->virtual_map->updateProbability(node->state, sensor_model_);
  }

  if (parameter_.verbose)
	std::cout << "  Done." << std::endl;
}

void EMPlanner2D::updateNodeInformation_EM(Node::shared_ptr node) {
  if (parameter_.verbose)
	std::cout << "  Update node information - EM" << std::endl;

  EMPlanner2D::Node::shared_ptr parent = node->parent.lock();
  Pose2 origin = parent->state.pose;
  Eigen::Matrix3d cov = parent->state.covariance();
  if (parameter_.dubins_control_model_enabled) {
	for (int step = 0; step < node->poses.size(); ++step) {
	  Pose2 odom = origin.between(node->poses[step]);

	  SimpleControlModel::ControlState
		  control_state = Simulator2D::move(origin, odom, control_model_, true, false);
//      Eigen::Matrix3d Q = control_state.getSigmas().asDiagonal();
//      Q = Q * Q;
//      cov = control_state.getFx1() * cov * control_state.getFx1().transpose() + Q;

	  unsigned int node_key = parent->key + step + 1;
	  unsigned int parent_key = parent->key + step;
	  SLAM2D::OdometryFactor2DPtr dubins_path_factor
		  = SLAM2D::buildOdometryFactor(parent_key, node_key, control_state);

	  node->odometry_factors.push_back(dubins_path_factor);

//      VehicleBeliefState state(node->poses[step]);
//      state.information = inverse(cov);
//      state.core_vehicle = (step == node->poses.size() - 1);
//      node->map->addVehicle(state);

	  origin = node->poses[step];
	}

	if (parameter_.verbose)
	  std::cout << "  Done with Dubins model." << std::endl;
  } else {
	const Pose2 &end = node->state.pose;
	Pose2 odom = origin.between(end);

	SimpleControlModel::ControlState
		control_state = Simulator2D::move(origin, odom, control_model_, true, false);
//    Eigen::Matrix3d Q = control_state.getSigmas().asDiagonal();
//    Q = Q * Q;
//    cov = control_state.getFx1() * cov * control_state.getFx1().transpose() + Q;

	unsigned int node_key = node->key;
	unsigned int parent_key = parent->key;

	SLAM2D::OdometryFactor2DPtr odometry_factor
		= SLAM2D::buildOdometryFactor(parent_key, node_key, control_state);
	node->odometry_factors.push_back(odometry_factor);

//    VehicleBeliefState state(end);
//    state.information = inverse(cov);
//    state.core_vehicle = true;
//    node->map->addVehicle(state);

	if (parameter_.verbose)
	  std::cout << "  Done with one step model." << std::endl;
  }

  node->state.information = inverse(cov);

  Simulator2D::MeasurementVector measurements
	  = Simulator2D::measure(*map_, node->state.pose, sensor_model_, false, false);

  if (measurements.size() == 0) {
//    node->virtual_map->updateInformation(node->state, sensor_model_);

	if (parameter_.verbose)
	  std::cout << "  No measurement and update information." << std::endl;
  } else {
	if (parameter_.verbose)
	  std::cout << "  Re-optimization with ISAM2" << std::endl;

	unsigned int node_key = node->key;
	for (auto &m : measurements) {
	  SLAM2D::MeasurementFactor2DPtr factor = SLAM2D::buildMeasurementFactor(node_key, m.first, m.second);
	  node->measurement_factors.push_back(factor);
	}

//    updateTrajectory_EM(node);
//    node->virtual_map->updateInformation(*node->map, sensor_model_);

	if (parameter_.verbose)
	  std::cout << "  Done and update information" << std::endl;
  }
}

void EMPlanner2D::updateNodeInformation_OG_SHANNON(Node::shared_ptr node) {
  if (parameter_.verbose)
	std::cout << "  Update node information - OG SHANNON" << std::endl;

  EMPlanner2D::Node::shared_ptr parent = node->parent.lock();
  Pose2 origin = parent->state.pose;
  if (parameter_.dubins_control_model_enabled) {
	for (int step = 0; step < node->poses.size(); ++step) {
	  Pose2 odom = origin.between(node->poses[step]);

	  SimpleControlModel::ControlState
		  control_state = Simulator2D::move(origin, odom, control_model_, false, false);

	  unsigned int node_key = parent->key + step + 1;
	  unsigned int parent_key = parent->key + step;
	  SLAM2D::OdometryFactor2DPtr dubins_path_factor
		  = SLAM2D::buildOdometryFactor(parent_key, node_key, control_state);

	  node->odometry_factors.push_back(dubins_path_factor);

	  VehicleBeliefState state(node->poses[step]);
	  state.core_vehicle = (step == node->poses.size() - 1);
	  node->map->addVehicle(state);

	  origin = node->poses[step];
	}

	if (parameter_.verbose)
	  std::cout << "  Done with Dubins model." << std::endl;
  } else {
	const Pose2 &end = node->state.pose;
	Pose2 odom = origin.between(end);

	SimpleControlModel::ControlState
		control_state = Simulator2D::move(origin, odom, control_model_, false, false);

	unsigned int node_key = node->key;
	unsigned int parent_key = parent->key;

	SLAM2D::OdometryFactor2DPtr odometry_factor
		= SLAM2D::buildOdometryFactor(parent_key, node_key, control_state);
	node->odometry_factors.push_back(odometry_factor);

	VehicleBeliefState state(end);
	state.core_vehicle = true;
	node->map->addVehicle(state);

	if (parameter_.verbose)
	  std::cout << "  Done with one step model." << std::endl;
  }

  Simulator2D::MeasurementVector measurements
	  = Simulator2D::measure(*map_, node->state.pose, sensor_model_, false, false);

  if (parameter_.verbose)
	std::cout << "  Re-optimization with ISAM2" << std::endl;

  unsigned int node_key = node->key;
  for (auto &m : measurements) {
	SLAM2D::MeasurementFactor2DPtr factor = SLAM2D::buildMeasurementFactor(node_key, m.first, m.second);
	node->measurement_factors.push_back(factor);
  }

  if (parameter_.algorithm == OptimizationAlgorithm::SLAM_OG_SHANNON)
	updateTrajectory_SLAM_OG_SHANNON(node);
  else if (parameter_.algorithm == OptimizationAlgorithm::OG_SHANNON)
	updateTrajectory_OG_SHANNON(node);
  else
	std::cout << "Wrong algorithm." << std::endl;

  if (parameter_.verbose)
	std::cout << "  Done and update information" << std::endl;
}

void EMPlanner2D::updateNode(EMPlanner2D::Node::shared_ptr node) {
  EMPlanner2D::Node::shared_ptr parent = node->parent.lock();

  if (parameter_.verbose) {
	std::cout << "Optimizing node" << std::endl;
	std::cout << "  Node: ";
	node->print();
	std::cout << "  Parent: ";
	parent->print();
  }

  updateNodeInformation(node);

#ifndef LEAFONLY
  node->uncertainty = calculateUncertainty(node);
  node->cost = costFunction(node);
#endif

  if (parameter_.verbose) {
	std::cout << "  Done: ";
	node->print();
  }
}

EMPlanner2D::OptimizationResult EMPlanner2D::rrt_planner(const SLAM2D &slam, const VirtualMap &virtual_map, int n_key, double fron_0, double fron_1) {
	initialize(slam, virtual_map);

	double safe_distance_backup = parameter_.safe_distance;
	// Decrease the safe distance if the vehicle is close to obstacles in the beginning.
	int nearest = map_->searchLandmarkNearest(map_->getCurrentVehicle());
	if (nearest < map_->getLandmarkSize()) {
		double distance = map_->getLandmark(nearest).point.distance(map_->getCurrentVehicle().pose.t());
		if (distance < parameter_.safe_distance) {
			parameter_.safe_distance = distance - 0.1;
		}
	}

	std::vector<gtsam::Key> all_key;
	gtsam::Key goal_key;
	double goal_x;
	double goal_y;

	if (n_key < slam_->key_size()) {
	  all_key = slam_->get_all_key();
	  goal_key = all_key[ n_key ];
	  goal_x = slam_->goal_x(goal_key);
	  goal_y = slam_->goal_y(goal_key);
	} else {
	  goal_x = fron_0;
	  goal_y = fron_1;
	}

	int num_nodes = 0;
	int failed = 0;
	bool reach_goal = false;

	Pose2 pose(goal_x, goal_y, M_PI);
	EMPlanner2D::Node::shared_ptr goal(new EMPlanner2D::Node(pose));
	while (!reach_goal) {
		if (parameter_.verbose)
			std::cout << "Start. sampling num: " << num_nodes << std::endl;

		EMPlanner2D::Node::shared_ptr node = sampleNode();
//        std::cout << "Start. sampling num: " << num_nodes << std::endl;
//        std::cout << "node_x: " << node->state.pose.x() << "node_y: " <<  node->state.pose.x()<< std::endl;
//        std::cout << "goal_x: " << goal->state.pose.x() << "goal_y: " <<  goal->state.pose.x()<< std::endl;



		if (node == nullptr) {
			parameter_.safe_distance = safe_distance_backup;
			return OptimizationResult::SAMPLING_FAILURE;
		}

		EMPlanner2D::Node::shared_ptr parent = nearestNode(node);

		if (!connectNode(node, parent)) {
			failed += 1;
			// std::cout << failed << std::endl;
			if (failed > 1000) {
				parameter_.safe_distance = safe_distance_backup;
				return OptimizationResult::SAMPLING_FAILURE;
			}
			continue;
		}
		failed = 0;

		nodes_.push_back(node);
		parent->children.push_back(node);

//        updateNode(node);

		if (parameter_.verbose) {
			std::cout << "Sample " << num_nodes << " done.";
			best_node_->print();
		}
		reach_goal = isReached(goal, node);

//        std::cout << "is reached: " << reach_goal << std::endl;
//
//        std::cout << "\n" << std::endl;
		if (reach_goal && connectNode(goal, node)){
		  nodes_.push_back(goal);
		  node->children.push_back(goal);
//          updateNode(goal);

		}
		num_nodes++;
	}

	best_node_ = goal;
	int vl_known = 0;
	for (auto it = best_node_->virtual_map->cbeginVirtualLandmark();
		 it != best_node_->virtual_map->cendVirtualLandnmark(); ++it)
		if (it->probability < parameter_.occupancy_threshold)
			vl_known++;

	double percentage = (double) vl_known / virtual_map.getVirtualLandmarkSize();
	update_distance_weight_ = true;
	parameter_.safe_distance = safe_distance_backup;
	return OptimizationResult::SUCCESS;
}

std::vector<Pose2> EMPlanner2D::line_planner(const SLAM2D &slam, const VirtualMap &virtual_map, int n_key, double fron_0, double fron_1) {
	initialize(slam, virtual_map);
	std::vector<Pose2> actions;
	double max_edge_distance = parameter_.max_edge_length;

	double  root_node_x = map_->getCurrentVehicle().pose.x();
	double  root_node_y = map_->getCurrentVehicle().pose.y();
	double  root_node_theta = map_->getCurrentVehicle().pose.theta();

	double safe_distance_backup = parameter_.safe_distance;
	// Decrease the safe distance if the vehicle is close to obstacles in the beginning.
	int nearest = map_->searchLandmarkNearest(map_->getCurrentVehicle());
	if (nearest < map_->getLandmarkSize()) {
		double distance = map_->getLandmark(nearest).point.distance(map_->getCurrentVehicle().pose.t());
		if (distance < parameter_.safe_distance) {
			parameter_.safe_distance = distance - 0.1;
		}
	}

	std::vector<gtsam::Key> all_key;
	gtsam::Key goal_key;
	double goal_x;
	double goal_y;
	double goal_theta;


	if (n_key < slam_->key_size()) {
	  all_key = slam_->get_all_key();
	  goal_key = all_key[ n_key ];
	  goal_x = slam_->goal_x(goal_key);
	  goal_y = slam_->goal_y(goal_key);
	} else {
	  goal_x = fron_0;
	  goal_y = fron_1;
	}
	goal_theta = atan2(goal_y - root_node_y, goal_x - root_node_x);

	if (root_node_theta < 0) {
	  root_node_theta = M_PI*2 + root_node_theta;
	}
	if (goal_theta < 0) {
	  goal_theta = M_PI * 2 + goal_theta;
	}

//  initial rotation
	double dr = 180*M_PI/180;

	double diff_theta = goal_theta - root_node_theta;
	if (diff_theta > M_PI){
	  diff_theta = 2*M_PI -diff_theta;

	  int quotient = int(diff_theta/dr);
	  double remainder = diff_theta - dr*quotient;
	  for( int i=0; i<quotient; i++){
		Pose2 step(0,0,-dr);
		actions.push_back(step);
	  }
	  Pose2 step(0,0,-remainder);
	  actions.push_back(step);
	} else if (diff_theta > -M_PI && diff_theta < 0){
	  diff_theta = std::abs(diff_theta);

	  int quotient = int(diff_theta/dr);
	  double remainder = diff_theta - dr*quotient;
	  for( int i=0; i<quotient; i++){
		Pose2 step(0,0,-dr);
		actions.push_back(step);
	  }

	  Pose2 step(0,0,-remainder);
	  actions.push_back(step);
	} else if (diff_theta <= -M_PI){
	  diff_theta = 2*M_PI - std::abs(diff_theta);
	  int quotient = int(diff_theta/dr);
	  double remainder = diff_theta - dr*quotient;
	  for( int i=0; i<quotient; i++){
		Pose2 step(0,0,dr);
		actions.push_back(step);
	  }
	  Pose2 step(0,0,remainder);
	  actions.push_back(step);
	} else{
	  int quotient = int(diff_theta/dr);
	  double remainder = diff_theta - dr*quotient;
	  for( int i=0; i<quotient; i++){
		Pose2 step(0,0,dr);
		actions.push_back(step);
	  }
	  Pose2 step(0,0,remainder);
	  actions.push_back(step);
	}

	double path_distance = sqrt(pow(root_node_x-goal_x,2)+pow(root_node_y-goal_y,2));
	int dist_quot = int(path_distance/max_edge_distance);
	double dist_remainder = path_distance - dist_quot*max_edge_distance;

	for ( int i=0; i<dist_quot; i++){
	  Pose2 step(max_edge_distance,0,0);
	  actions.push_back(step);
	}
	Pose2 step(dist_remainder,0,0);
	actions.push_back(step);

	return actions;
}

EMPlanner2D::OptimizationResult EMPlanner2D::optimize2(const SLAM2D &slam, const VirtualMap &virtual_map) {
  initialize(slam, virtual_map);

  double safe_distance_backup = parameter_.safe_distance;
  // Decrease the safe distance if the vehicle is close to obstacles in the beginning.
  int nearest = map_->searchLandmarkNearest(map_->getCurrentVehicle());
  if (nearest < map_->getLandmarkSize()) {
	double distance = map_->getLandmark(nearest).point.distance(map_->getCurrentVehicle().pose.t());
	if (distance < parameter_.safe_distance) {
	  parameter_.safe_distance = std::max(0.0, distance - 0.1);
	}
  }

  int num_nodes = 0;
  int failed = 0;
  while (num_nodes < max_nodes_) {
	if (parameter_.verbose)
	  std::cout << "Start. sampling num: " << num_nodes << std::endl;

	EMPlanner2D::Node::shared_ptr node = sampleNode();
	if (node == nullptr) {
	  parameter_.safe_distance = safe_distance_backup;
	  return OptimizationResult::SAMPLING_FAILURE;
	}

	EMPlanner2D::Node::shared_ptr parent = nearestNode(node);

	if (!connectNode(node, parent)) {
	  failed += 1;
	  // std::cout << failed << std::endl;
	  if (failed > 1000) {
		parameter_.safe_distance = safe_distance_backup;
		return OptimizationResult::SAMPLING_FAILURE;
	  }
	  continue;
	}
	failed = 0;

	nodes_.push_back(node);
	parent->children.push_back(node);

	updateNode(node);

	if (parameter_.verbose) {
	  std::cout << "Sample " << num_nodes << " done.";
	  best_node_->print();
	}

	num_nodes++;
  }

//  std::cout << "root: " << root_->distance << ", " << root_->uncertainty << ", " << distance_weight_ << ", " << root_->cost << std::endl;
  for (Node::shared_ptr node : nodes_) {
	if (node->children.size() != 0)
	  continue;

	updateTrajectory_EM(node);
	node->virtual_map.reset(new VirtualMap(*root_->virtual_map));
	node->virtual_map->updateInformation(*node->map, sensor_model_);

	node->uncertainty = calculateUncertainty(node);
	node->cost = costFunction(node);

	if (parameter_.algorithm == OptimizationAlgorithm::SLAM_OG_SHANNON && best_node_ == root_) {
	  best_node_ = node;
	}

	if (best_node_ == root_ || node->cost < best_node_->cost)
	  best_node_ = node;

//    std::cout << "node: " << node->distance << ", " << node->uncertainty << ", " << distance_weight_ << ", " << node->cost << std::endl;
  }

  int vl_known = 0;
  for (auto it = best_node_->virtual_map->cbeginVirtualLandmark();
	   it != best_node_->virtual_map->cendVirtualLandnmark(); ++it)
	if (it->probability < parameter_.occupancy_threshold)
	  vl_known++;

  double percentage = (double) vl_known / virtual_map.getVirtualLandmarkSize();
  // std::cout << "Map coverage: " << percentage << std::endl;

  update_distance_weight_ = true;
  parameter_.safe_distance = safe_distance_backup;
  return OptimizationResult::SUCCESS;
}

EMPlanner2D::OptimizationResult EMPlanner2D::optimize(const SLAM2D &slam, const VirtualMap &virtual_map) {
  initialize(slam, virtual_map);
  // assert(parameter_.safe_distance > 0);
  double safe_distance_backup = parameter_.safe_distance;
  // Decrease the safe distance if the vehicle is close to obstacles in the beginning.
  int nearest = map_->searchLandmarkNearest(map_->getCurrentVehicle());
  if (nearest < map_->getLandmarkSize()) {
	double distance = map_->getLandmark(nearest).point.distance(map_->getCurrentVehicle().pose.t());
	if (distance < parameter_.safe_distance) {
	  parameter_.safe_distance = std::max(0.1, distance - 0.1);
	}
  }

  int num_nodes = 0;
  int failed = 0;
  while (num_nodes < max_nodes_) {
	if (parameter_.verbose)
	  std::cout << "Start. sampling num: " << num_nodes << std::endl;

	EMPlanner2D::Node::shared_ptr node = sampleNode();
	if (node == nullptr) {
	  parameter_.safe_distance = safe_distance_backup;
	  return OptimizationResult::SAMPLING_FAILURE;
	}

	EMPlanner2D::Node::shared_ptr parent = nearestNode(node);

	if (!connectNode(node, parent)) {
	  failed += 1;
	  // std::cout << failed << std::endl;
	  if (failed > 1000) {
		parameter_.safe_distance = safe_distance_backup;
		return OptimizationResult::SAMPLING_FAILURE;
	  }
	  continue;
	}
	failed = 0;

	nodes_.push_back(node);
	parent->children.push_back(node);

	updateNode(node);

#ifndef LEAFONLY
	if (node->cost < best_node_->cost) {
	  best_node_ = node;
	}
#endif

	if (parameter_.verbose) {
	  std::cout << "Sample " << num_nodes << " done.";
	  best_node_->print();
	}

	num_nodes++;
  }

#ifdef LEAFONLY
  best_node_->cost = 1e10;
  for (Node::shared_ptr node : nodes_) {
	if (node->children.size() != 0)
	  continue;

	node->uncertainty = calculateUncertainty(node);
	node->cost = costFunction(node);
	// std::cout << "uncertainty: " << node->uncertainty << ", distance: " << node->distance << std::endl;

	if (parameter_.algorithm == OptimizationAlgorithm::SLAM_OG_SHANNON && best_node_ == root_) {
	  best_node_ = node;
	}

	if (node->cost < best_node_->cost)
	  best_node_ = node;
  }
#endif

  int vl_known = 0;
  for (auto it = best_node_->virtual_map->cbeginVirtualLandmark();
	   it != best_node_->virtual_map->cendVirtualLandnmark(); ++it)
	if (it->probability < parameter_.occupancy_threshold)
	  vl_known++;

  double percentage = (double) vl_known / virtual_map.getVirtualLandmarkSize();
  // std::cout << "Map coverage: " << percentage << std::endl;
  // best_node_->print();
  parameter_.safe_distance = safe_distance_backup;

  if (best_node_ == root_) {
	return OptimizationResult::NO_SOLUTION;

	if (percentage < 0.95) {
	  if (fabs(distance_weight_) < 1e-5) {
		std::cout << "Failed to find the best path. Stop because the distance weight is zero." << std::endl;
		update_distance_weight_ = true;
		return OptimizationResult::NO_SOLUTION;
	  }

//      while (true) {
//        double dw = parameter_.distance_weight0 * parameter_.d_weight;
//        distance_weight_ = (distance_weight_ - dw < 0) ? 0.0 : distance_weight_ - dw;
//        if (fabs(distance_weight_) < 1e-5) {
//          std::cout << "Failed to find the best path. Stop because the distance weight is zero." << std::endl;
//          update_distance_weight_ = true;
//          return OptimizationResult::NO_SOLUTION;
//        }
//        for (Node::shared_ptr node : nodes_) {
//          if (node->children.size() != 0)
//            continue;
//
//          node->cost = costFunction(node);
//
//          if (node->cost < best_node_->cost)
//            best_node_ = node;
//        }
//        if (best_node_ != root_)
//          return OptimizationResult::SUCCESS;
//      }

	  double dw = parameter_.distance_weight0 * parameter_.d_weight;
	  distance_weight_ = (distance_weight_ - dw < 0) ? 0.0 : distance_weight_ - dw;
	  if (fabs(distance_weight_) < 1e-5) {
		std::cout << "Failed to find the best path. Stop because the distance weight is zero." << std::endl;
		update_distance_weight_ = true;
		return OptimizationResult::NO_SOLUTION;
	  }
	  for (Node::shared_ptr node : nodes_) {
		if (node->children.size() != 0)
		  continue;

		node->cost = costFunction(node);

		if (node->cost < best_node_->cost)
		  best_node_ = node;
	  }
	  if (best_node_ != root_) {
//        parameter_.safe_distance = safe_distance_backup;
		return OptimizationResult::SUCCESS;
	  } else {
		update_distance_weight_ = false;
//        parameter_.safe_distance = safe_distance_backup;
		return optimize(slam, virtual_map);
	  }
	} else {
	  return OptimizationResult::TERMINATION;
	}
  }
  update_distance_weight_ = true;
  parameter_.safe_distance = safe_distance_backup;
  return OptimizationResult::SUCCESS;
}

void EMPlanner2D::initialize(const SLAM2D &slam, const VirtualMap &virtual_map) {
  if (parameter_.verbose)
	std::cout << "Initialize planner." << std::endl;

  slam_ = &slam;
  map_ = &slam.getMap();
  virtual_map_ = &virtual_map;
  values_.reset(new gtsam::Values(slam.getISAM2()->calculateBestEstimate()));
  for (int i = 0; i < map_->getTrajectorySize(); ++i) {
	const auto &v = map_->getVehicle(i);
	if (v.core_vehicle)
	  updated_keys_.insert(SLAM2D::getVehicleSymbol(i).key());
  }

  root_.reset(new EMPlanner2D::Node(map_->getCurrentVehicle()));
  assert(root_->state.core_vehicle);
  root_->key = static_cast<unsigned int>(map_->getTrajectorySize() - 1);

  root_->map.reset(new Map(*map_));
  root_->virtual_map.reset(new VirtualMap(virtual_map));
  const gtsam::NonlinearFactorGraph &graph = slam.getISAM2()->getFactorsUnsafe();
  root_->measurement_factors.resize(graph.size());
  for (int i = 0; i < graph.size(); ++i)
	root_->measurement_factors[i] = graph.at(i);

  //////////////////////////
  gtsam::ISAM2Params params;
  params.enableRelinearization = false;
  root_->isam.reset(new gtsam::ISAM2(params));
  root_->isam->update(graph, *values_);
  //////////////////////////

  root_->children.clear();
  root_->distance = 0.0;
  root_->n_dubins = -1;
  root_->uncertainty = calculateUncertainty(root_);

  best_node_ = root_;
  nodes_.clear();
  nodes_.push_back(root_);

  int vl_known = 0;
  for (auto it = virtual_map_->cbeginVirtualLandmark(); it != virtual_map_->cendVirtualLandnmark(); ++it)
	if (it->probability < parameter_.occupancy_threshold)
	  vl_known++;

  max_nodes_ = static_cast<int>(floor(vl_known * parameter_.max_nodes));
  double percentage_known = (double) vl_known / virtual_map_->getVirtualLandmarkSize();

//  if (update_distance_weight_) {
	distance_weight_ = parameter_.distance_weight0
		- (parameter_.distance_weight0 - parameter_.distance_weight1) * percentage_known;
//  }

  if (parameter_.verbose) {
    std::cout << "  Graph size: " << root_->measurement_factors.size() << std::endl;
	std::cout << "  Percentage known: " << percentage_known << std::endl;
	std::cout << "  Distance weight: " << distance_weight_ << std::endl;
  }

  if (parameter_.algorithm == OptimizationAlgorithm::SLAM_OG_SHANNON) {
	double og_entropy = 0.0;
	for (auto it = virtual_map_->cbeginVirtualLandmark(); it != virtual_map_->cendVirtualLandnmark(); ++it) {
	  double p = it->probability;
	  og_entropy += -p * log(p) - (1 - p) * log(1 - p);
	}
	w2_ = (1 - parameter_.alpha) / og_entropy;

	double slam_uncertainty = 0.0;
	for (auto it = map_->cbeginLandmark(); it != map_->cendLandmark(); ++it) {
	  slam_uncertainty += sqrt(it->second.covariance().determinant());
	}
	w1_ = parameter_.alpha / slam_uncertainty;
  }

  root_->cost = costFunction(root_);
}

void EMPlanner2D::initializeDubinsPathLibrary() {
  if (parameter_.verbose)
	std::cout << "Initialize Dubins path library" << std::endl;

  assert(parameter_.dubins_parameter.max_w > 0);
  dubins_library_.clear();

  for (double v = parameter_.dubins_parameter.max_v; v > parameter_.dubins_parameter.min_v - 1e-10;
	   v -= parameter_.dubins_parameter.dv) {
	for (double w = 0; w < parameter_.dubins_parameter.max_w + 1e-10;
		 w += parameter_.dubins_parameter.dw) {
	  for (int s = -1; s <= 1; s += 2) {
		double t = 0.0;
		double ww = w * s;

		/*
		std::vector<Pose2> poses;
		poses.push_back(Pose2(0, 0, 0));
		while (t < parameter_.dubins_parameter.max_duration) {
		  const Pose2 &pose = poses.back();

		  double x = pose.x() + v * parameter_.dubins_parameter.dt * pose.r().c();
		  double y = pose.y() + v * parameter_.dubins_parameter.dt * pose.r().s();
		  double theta = pose.theta() + ww * parameter_.dubins_parameter.dt;
		  poses.emplace_back(x, y, theta);
		  t += parameter_.dubins_parameter.dt;
		}
		dubins_library_.push_back(poses);
		*/

		Pose2 pose(0, 0, 0);
		Dubins dubins;
		dubins.v = v;
		dubins.w = ww;
		dubins.num_steps = 0;
		while (t < parameter_.dubins_parameter.max_duration) {
		  dubins.num_steps++;

		  double x = pose.x() + dubins.v * parameter_.dubins_parameter.dt * pose.r().c();
		  double y = pose.y() + dubins.v * parameter_.dubins_parameter.dt * pose.r().s();
		  double theta = pose.theta() + dubins.w * parameter_.dubins_parameter.dt;
		  pose = Pose2(x, y, theta);
		  t += parameter_.dubins_parameter.dt;

		  if (t > parameter_.dubins_parameter.min_duration) {
			dubins.end = pose;
			dubins_library_.push_back(dubins);
		  }
		}
	  }
	}
  }

  if (parameter_.verbose)
	std::cout << "  Done. Library size: " << dubins_library_.size() << std::endl;
}

double EMPlanner2D::simulations_reward(const SLAM2D &slam,const VirtualMap &virtual_map,const Simulator2D &sim, std::vector<Pose2> actions){
    std::shared_ptr<SLAM2D> temp_slam(new SLAM2D(slam));
	temp_slam->set_copy_isam();
    std::shared_ptr<VirtualMap> temp_virtual_map(new VirtualMap(virtual_map));
    std::shared_ptr<Simulator2D> temp_sim(new Simulator2D(sim));

    double angle_weight = parameter_.angle_weight;
    bool obstacle;
    double dist=0;

    double initial_u;
    double final_u;
    double reward;
    initial_u = EMPlanner2D::calculateUtility(*temp_virtual_map, 0, parameter_);

    int actions_size = int(actions.size());
    for (int i=0; i < actions_size; i++){
        Pose2 step_odom = actions[i];
        double step_x = step_odom.x();
        double step_y = step_odom.y();
        double step_theta = step_odom.theta();
        std::pair<bool, SimpleControlModel::ControlState> move_result;
        dist = dist + sqrt(pow(step_x, 2)+pow(step_y, 2) + angle_weight * pow(step_theta, 2));
//        std::cout<< "111" << std::endl;

//        move fuction
        move_result = temp_sim->move(step_odom, true);

//        std::cout<< "112" << std::endl;
        temp_slam->addOdometry(move_result.second);

//        std::cout<< "222" << std::endl;
//        measure fuction
        Simulator2D::MeasurementVector measure_result;
        measure_result = temp_sim->measure();

        int key_size = int(measure_result.size());
        for (int j=0; j<key_size; j++){
            temp_slam->addMeasurement(measure_result[j].first, measure_result[j].second);
            }

        temp_slam->copy_optimize(true);
        temp_virtual_map->updateProbability(*temp_slam, temp_sim->getSensorModel());
        temp_virtual_map->updateInformation(temp_slam->getMap(), temp_sim->getSensorModel());
        }

//    get reward of the current actions
    final_u = EMPlanner2D::calculateUtility(*temp_virtual_map, dist, parameter_);
    reward = initial_u - final_u;
//    std::cout<< "reward: " << reward << std::endl;
    return reward;

    }


}
