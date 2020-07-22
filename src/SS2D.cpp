#include "em_exploration/Simulation2D.h"
#include "em_exploration/SLAM2D.h"
#include "em_exploration/VirtualMap.h"
#include "em_exploration/OccupancyMap.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#include <pybind11/operators.h>

using namespace gtsam;
using namespace em_exploration;

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(Point2);

PYBIND11_MODULE(ss2d, m) {
  m.doc() = "2D simulation and SLAM module";

  py::class_<Pose2>(m, "Pose2")
      .def(py::init<double, double, double>())
      .def_property_readonly("x", &Pose2::x)
      .def_property_readonly("y", &Pose2::y)
      .def_property_readonly("theta", &Pose2::theta)
      .def("__mul__", [](const Pose2 &a, const Pose2 &b) {
        return a * b;
      }, py::is_operator())
      .def("between", [](const Pose2 &pose1, const Pose2 &pose2) -> std::tuple<Pose2, Matrix, Matrix> {
        Eigen::Matrix3d H1, H2;
        Pose2 pose = pose1.between(pose2, H1, H2);
        return std::make_tuple(pose, H1, H2);
      })
      .def("__repr__", [](const Pose2 &p) { return
        "[" + std::to_string(p.x()) + ", " +
              std::to_string(p.y()) + ", " +
              std::to_string(p.theta()) + "]"; });

  py::class_<Point2>(m, "Point2")
      .def(py::init<double, double>())
      .def_property_readonly("x", &Point2::x)
      .def_property_readonly("y", &Point2::y)
      .def("__repr__", [](const Point2 &p) { return
        "[" + std::to_string(p.x()) + ", " +
              std::to_string(p.y()) + "]"; });

  py::class_<Rot2>(m, "Rot2")
      .def(py::init<double>())
      .def_property_readonly("theta", &Rot2::theta)
      .def("__repr__", [](const Rot2 &r) { return
        "[" + std::to_string(r.theta()) + "]"; });

  py::class_<BearingRangeSensorModel::Parameter>(m, "BearingRangeSensorModelParameter")
      .def(py::init<>())
      .def_property("bearing_noise",
                    &BearingRangeSensorModel::Parameter::getBearingNoise,
                    &BearingRangeSensorModel::Parameter::setBearingNoise)
      .def_property("range_noise", &BearingRangeSensorModel::Parameter::getRangeNoise,
      &BearingRangeSensorModel::Parameter::setRangeNoise)
      .def_property("min_bearing",
                    &BearingRangeSensorModel::Parameter::getMinBearing,
                    &BearingRangeSensorModel::Parameter::setMinBearing)
      .def_property("max_bearing",
                    &BearingRangeSensorModel::Parameter::getMaxBearing,
                    &BearingRangeSensorModel::Parameter::setMaxBearing)
      .def_property("min_range", &BearingRangeSensorModel::Parameter::getMinRange,
      &BearingRangeSensorModel::Parameter::setMinRange)
      .def_property("max_range", &BearingRangeSensorModel::Parameter::getMaxRange,
      &BearingRangeSensorModel::Parameter::setMaxRange)
      .def("pprint", &BearingRangeSensorModel::Parameter::print);

  py::class_<BearingRangeSensorModel::Measurement>(m, "BearingRangeSensorModelMeasurement")
      .def_property_readonly("has_jacobian", &BearingRangeSensorModel::Measurement::hasJacobian)
      .def_property_readonly("bearing", &BearingRangeSensorModel::Measurement::getBearing)
      .def_property_readonly("range", &BearingRangeSensorModel::Measurement::getRange)
      .def_property_readonly("sigmas", &BearingRangeSensorModel::Measurement::getSigmas)
      .def_property_readonly("Hx", &BearingRangeSensorModel::Measurement::getHx)
      .def_property_readonly("Hl", &BearingRangeSensorModel::Measurement::getHl)
      .def("transform_from", &BearingRangeSensorModel::Measurement::transformFrom)
      .def("pprint", &BearingRangeSensorModel::Measurement::print);

  py::class_<BearingRangeSensorModel>(m, "BearingRangeSensorModel")
      .def(py::init<const BearingRangeSensorModel::Parameter &>())
      .def(py::init<const BearingRangeSensorModel::Parameter &, RNG::SeedType>())
      .def_property_readonly("parameter", &BearingRangeSensorModel::getParameter)
      .def("check_validity", &BearingRangeSensorModel::check)
      .def("measure", &BearingRangeSensorModel::measure);

  py::class_<SimpleControlModel::Parameter>(m, "SimpleControlModelParameter")
      .def(py::init<>())
      .def_property("translation_noise", &SimpleControlModel::Parameter::getTranslationNoise, &SimpleControlModel::Parameter::setTranslationNoise)
      .def_property("rotation_noise",
                    &SimpleControlModel::Parameter::getRotationNoise,
                    &SimpleControlModel::Parameter::setRotationNoise)
      .def("pprint", &SimpleControlModel::Parameter::print);

  py::class_<SimpleControlModel::ControlState>(m, "SimpleControlModelState")
      .def(py::init<>())
      .def_property_readonly("has_jacobian", &SimpleControlModel::ControlState::hasJacobian)
      .def_property_readonly("pose", &SimpleControlModel::ControlState::getPose)
      .def_property_readonly("odom", &SimpleControlModel::ControlState::getOdom)
      .def_property_readonly("sigmas", &SimpleControlModel::ControlState::getSigmas)
      .def_property_readonly("Fx1", &SimpleControlModel::ControlState::getFx1)
      .def_property_readonly("Fx2", &SimpleControlModel::ControlState::getFx2)
      .def("pprint", &SimpleControlModel::ControlState::print);

  py::class_<SimpleControlModel>(m, "SimpleControlModel")
      .def(py::init<const SimpleControlModel::Parameter &>())
      .def(py::init<const SimpleControlModel::Parameter &, RNG::SeedType>())
      .def_property_readonly("parameter", &SimpleControlModel::getParameter)
      .def("evolve", &SimpleControlModel::evolve);

  py::class_<VehicleBeliefState>(m, "VehicleBeliefState")
      .def(py::init<>())
      .def(py::init<const Pose2 &>())
      .def(py::init<const Pose2 &, const Eigen::Matrix3d &>())
      .def_readwrite("core_vehicle", &VehicleBeliefState::core_vehicle)
      .def_readwrite("pose", &VehicleBeliefState::pose)
      .def_readwrite("information", &VehicleBeliefState::information)
      .def_property_readonly("covariance", &VehicleBeliefState::covariance)
      .def_property_readonly("global_covariance", &VehicleBeliefState::globalCovariance);

  py::class_<LandmarkBeliefState>(m, "LandmarkBeliefState")
      .def(py::init<>())
      .def(py::init<const Point2 &>())
      .def(py::init<const Point2 &, const Eigen::Matrix2d &>())
      .def_readwrite("point", &LandmarkBeliefState::point)
      .def_readwrite("information", &LandmarkBeliefState::information)
      .def_property_readonly("covariance", &LandmarkBeliefState::covariance);

  py::class_<Environment::Parameter>(m, "EnvironmentParameter")
      .def(py::init<>())
      .def_property("min_x", &Environment::Parameter::getMinX, &Environment::Parameter::setMinX)
      .def_property("min_y", &Environment::Parameter::getMinY, &Environment::Parameter::setMinY)
      .def_property("max_x", &Environment::Parameter::getMaxX, &Environment::Parameter::setMaxX)
      .def_property("max_y", &Environment::Parameter::getMaxY, &Environment::Parameter::setMaxY)
      .def_readwrite("max_steps", &Environment::Parameter::max_steps)
      .def_property("safe_distance", &Environment::Parameter::getSafeDistance, &Environment::Parameter::setSafeDistance)
      .def("pprint", &Environment::Parameter::print);

  py::class_<Environment, std::shared_ptr<Environment>>(m, "Environment")
      .def(py::init<const Environment::Parameter &>())
      .def_property_readonly("parameter", &Environment::getParameter)
      .def_property_readonly("distance", &Environment::getDistance)
      .def("add_landmark", (void (Environment::*)(unsigned int, const Point2 &)) &Environment::addLandmark)
      .def("add_landmark", (void (Environment::*)(unsigned int, const LandmarkBeliefState &)) &Environment::addLandmark)
      .def("update_landmark",
           (void (Environment::*)(unsigned int, const LandmarkBeliefState &)) &Environment::updateLandmark)
      .def("get_landmark", &Environment::getLandmark)
      .def("add_vehicle", (void (Environment::*)(const Pose2 &)) &Environment::addVehicle)
      .def("add_vehicle", (void (Environment::*)(const VehicleBeliefState &)) &Environment::addVehicle)
      .def("update_vehicle",
           (void (Environment::*)(unsigned int, const VehicleBeliefState &)) &Environment::updateVehicle)
      .def("get_vehicle", &Environment::getVehicle)
      .def("get_current_vehicle", &Environment::getCurrentVehicle)
      .def("clear", &Environment::clear)
      .def("search_landmark_neighbors", &Environment::searchLandmarkNeighbors)
      .def("search_trajectory_neighbors", &Environment::searchTrajectoryNeighbors)
      .def("search_landmark_nearest", &Environment::searchLandmarkNearest)
      .def("search_trajectory_nearest", &Environment::searchTrajectoryNearest)
      .def("check_safety", &Environment::checkSafety)
      .def("get_landmark_size", &Environment::getLandmarkSize)
      .def("get_trajectory_size", &Environment::getTrajectorySize)
      .def("iter_landmarks", [](Environment &env) {
        return py::make_iterator(env.beginLandmark(), env.endLandmark());
      }, py::keep_alive<0, 1>())
      .def("iter_trajectory", [](Environment &env) {
        return py::make_iterator(env.beginTrajectory(), env.endTrajectory());
      });

  typedef Environment Map;

  py::class_<Simulator2D>(m, "Simulator2D")
      .def(py::init<const BearingRangeSensorModel::Parameter &,
                    const SimpleControlModel::Parameter &>())
      .def(py::init<const BearingRangeSensorModel::Parameter &,
                    const SimpleControlModel::Parameter &,
                    RNG::SeedType>())
      .def("random_landmarks", &Simulator2D::addLandmarks)
      .def("pprint", &Simulator2D::printParameters)
      .def_property_readonly("vehicle", &Simulator2D::getVehicleState)
      .def("initialize_vehicle", &Simulator2D::initializeVehicleState)
      .def("move", (std::pair<bool, SimpleControlModel::ControlState> (Simulator2D::*)(const Pose2 &, bool))&Simulator2D::move)
      .def("measure", (Simulator2D::MeasurementVector (Simulator2D::*)() const)&Simulator2D::measure)
      .def_property_readonly("environment", &Simulator2D::getEnvironment)
      .def_property_readonly("sensor_model", &Simulator2D::getSensorModel)
      .def_property_readonly("control_model", &Simulator2D::getControlModel);

  py::class_<SLAM2D>(m, "SLAM2D")
      .def(py::init<const Map::Parameter&>())
      .def("pprint", &SLAM2D::printParameters)
      .def("add_prior", (void (SLAM2D::*)(unsigned int, const LandmarkBeliefState &)) &SLAM2D::addPrior)
      .def("add_prior", (void (SLAM2D::*)(const VehicleBeliefState &))&SLAM2D::addPrior)
      .def("add_odometry", &SLAM2D::addOdometry)
      .def("add_measurement", (void (SLAM2D::*)(unsigned int, const BearingRangeSensorModel::Measurement&))&SLAM2D::addMeasurement)
      .def("add_measurement", (void (SLAM2D::*)(unsigned int, const BearingRangeSensorModel::Measurement&, double))&SLAM2D::addMeasurement)
      .def("save_graph", &SLAM2D::saveGraph)
      .def("print_graph", &SLAM2D::printGraph)
      .def("optimize", &SLAM2D::optimize, py::arg("update_covariance") = true)
      .def("joint_marginal_covariance", &SLAM2D::jointMarginalCovariance)
      .def("joint_marginal_covariance_local", &SLAM2D::jointMarginalCovarianceLocal)
      .def("adjacency_degree_get", &SLAM2D::adjacency_degree_get)
      .def("adjacency_out", &SLAM2D::adjacency_out)
      .def("features_out", &SLAM2D::features_out)
      .def("get_key_points", &SLAM2D::get_key_points)
      .def("key_size", &SLAM2D::key_size)
      .def("sample", &SLAM2D::sample)
      .def_property_readonly("map", &SLAM2D::getMap);

  py::class_<VirtualMap::Parameter, Map::Parameter>(m, "VirtualMapParameter")
      .def(py::init<const Map::Parameter &>())
      .def_property("sigma0", &VirtualMap::Parameter::getSigma0, &VirtualMap::Parameter::setSigma0)
      .def_property("resolution", &VirtualMap::Parameter::getResolution, &VirtualMap::Parameter::setResolution)
      .def_property("num_samples", &VirtualMap::Parameter::getNumSamples, &VirtualMap::Parameter::setNumSamples)
      .def("pprint", &VirtualMap::Parameter::print);

  py::class_<VirtualMap::VirtualLandmark>(m, "VirtualLandmark")
      .def(py::init<>())
      .def_readwrite("updated", &VirtualMap::VirtualLandmark::updated)
      .def_readwrite("probability", &VirtualMap::VirtualLandmark::probability)
      .def_readwrite("point", &VirtualMap::VirtualLandmark::point)
      .def_readwrite("information", &VirtualMap::VirtualLandmark::information)
      .def_property_readonly("covariance", &VirtualMap::VirtualLandmark::covariance);

  py::class_<VirtualMap>(m, "VirtualMap")
      .def(py::init<const VirtualMap::Parameter &, RNG::SeedType>())
      .def("get_parameter", &VirtualMap::getParameter)
      .def("get_virtual_landmark_size", &VirtualMap::getVirtualLandmarkSize)
      .def("explored", &VirtualMap::explored)
      .def("update_probability", (void (VirtualMap::*)(const SLAM2D &, const BearingRangeSensorModel &))&VirtualMap::updateProbability)
      .def("update_information", (void (VirtualMap::*)(VirtualMap::VirtualLandmark &,
                                                       const Map &,
                                                       const BearingRangeSensorModel &) const)&VirtualMap::updateInformation)
      .def("update_information", (void (VirtualMap::*)(const Map &,
                                                       const BearingRangeSensorModel &))&VirtualMap::updateInformation)
      .def_property_readonly("rows", &VirtualMap::rows)
      .def_property_readonly("cols", &VirtualMap::cols)
      .def("to_array", &VirtualMap::toArray)
      .def("to_cov_array", &VirtualMap::toCovArray)
      .def("to_cov_trace", &VirtualMap::toCovTrace)
      .def("iter_virtual_landmarks", [](const VirtualMap &map) {
        return py::make_iterator(map.cbeginVirtualLandmark(), map.cendVirtualLandnmark()); })
      .def("get_sampled_map_size", &VirtualMap::getSampledMapSize)
      .def("get_sampled_map", &VirtualMap::getSampledMap);

  py::class_<OccupancyMap::Parameter, Map::Parameter>(m, "OccupancyMapParameter")
      .def(py::init<const Map::Parameter &>())
      .def_property("resolution", &OccupancyMap::Parameter::getResolution, &OccupancyMap::Parameter::setResolution)
      .def("pprint", &OccupancyMap::Parameter::print);

  py::class_<OccupancyMap>(m, "OccupancyMap")
      .def(py::init<const OccupancyMap::Parameter &>())
      .def("get_parameter", &OccupancyMap::getParameter)
      .def("to_array", &OccupancyMap::toArray)
      .def("update", (void (OccupancyMap::*)(int, int, bool))&OccupancyMap::update)
      .def("update", (void (OccupancyMap::*)(const Map &, const BearingRangeSensorModel &))&OccupancyMap::update)
      .def("update", (void (OccupancyMap::*)(const VehicleBeliefState &, const BearingRangeSensorModel &))&OccupancyMap::update);
}

