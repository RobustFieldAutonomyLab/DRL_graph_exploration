#include "em_exploration/Planner2D.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace em_exploration;
namespace py = pybind11;

PYBIND11_MODULE(planner2d, m) {
  m.doc() = "planner2d", "2D planner module";

  py::class_<EMPlanner2D::DubinsParameter>(m, "DubinsParameter")
      .def(py::init<>())
      .def_readwrite("max_w", &EMPlanner2D::DubinsParameter::max_w)
      .def_readwrite("dw", &EMPlanner2D::DubinsParameter::dw)
      .def_readwrite("min_v", &EMPlanner2D::DubinsParameter::min_v)
      .def_readwrite("max_v", &EMPlanner2D::DubinsParameter::max_v)
      .def_readwrite("dv", &EMPlanner2D::DubinsParameter::dv)
      .def_readwrite("dt", &EMPlanner2D::DubinsParameter::dt)
      .def_readwrite("min_duration", &EMPlanner2D::DubinsParameter::min_duration)
      .def_readwrite("max_duration", &EMPlanner2D::DubinsParameter::max_duration)
      .def_readwrite("tolerance_radius", &EMPlanner2D::DubinsParameter::tolerance_radius);

  py::class_<EMPlanner2D::Parameter>(m, "EMPlannerParameter")
      .def(py::init())
      .def_readwrite("verbose", &EMPlanner2D::Parameter::verbose)
      .def_readwrite("seed", &EMPlanner2D::Parameter::seed)
      .def_readwrite("max_edge_length", &EMPlanner2D::Parameter::max_edge_length)
      .def_readwrite("num_actions", &EMPlanner2D::Parameter::num_actions)
      .def_readwrite("max_nodes", &EMPlanner2D::Parameter::max_nodes)
      .def_readwrite("angle_weight", &EMPlanner2D::Parameter::angle_weight)
      .def_readwrite("distance_weight0", &EMPlanner2D::Parameter::distance_weight0)
      .def_readwrite("distance_weight1", &EMPlanner2D::Parameter::distance_weight1)
      .def_readwrite("d_weight", &EMPlanner2D::Parameter::d_weight)
      .def_readwrite("occupancy_threshold", &EMPlanner2D::Parameter::occupancy_threshold)
      .def_readwrite("safe_distance", &EMPlanner2D::Parameter::safe_distance)
      .def_readwrite("alpha", &EMPlanner2D::Parameter::alpha)
      .def_readwrite("algorithm", &EMPlanner2D::Parameter::algorithm)
      .def_readwrite("dubins_control_model_enabled", &EMPlanner2D::Parameter::dubins_control_model_enabled)
      .def_readwrite("dubins_parameter", &EMPlanner2D::Parameter::dubins_parameter)
      .def_readwrite("reg_out", &EMPlanner2D::Parameter::reg_out)
      .def("pprint", &EMPlanner2D::Parameter::print);

  py::class_<EMPlanner2D::Dubins>(m, "Dubins")
      .def(py::init<>())
      .def_readonly("v", &EMPlanner2D::Dubins::v)
      .def_readonly("w", &EMPlanner2D::Dubins::w)
      .def_readonly("end", &EMPlanner2D::Dubins::end)
      .def_readonly("num_steps", &EMPlanner2D::Dubins::num_steps);

  py::class_<EMPlanner2D::Node>(m, "Node")
      .def(py::init<const VehicleBeliefState &>())
      .def_readonly("key", &EMPlanner2D::Node::key)
      .def_readonly("state", &EMPlanner2D::Node::state)
      .def_readonly("poses", &EMPlanner2D::Node::poses)
      .def_readonly("distance", &EMPlanner2D::Node::distance)
      .def_readonly("uncertainty", &EMPlanner2D::Node::uncertainty)
      .def_readonly("cost", &EMPlanner2D::Node::cost)
      .def_property_readonly("map", &EMPlanner2D::Node::getMap)
      .def_property_readonly("virtual_map", &EMPlanner2D::Node::getVirtualMap);

  py::class_<EMPlanner2D::Edge>(m, "Edge")
      .def_property_readonly("first", &EMPlanner2D::Edge::getFirst)
      .def_property_readonly("second", &EMPlanner2D::Edge::getSecond)
      .def("get_odoms", &EMPlanner2D::Edge::getOdoms);

  py::class_<EMPlanner2D::ForwardIterator>(m, "EMPlannerForwardIterator")
      .def_property_readonly("edge", &EMPlanner2D::ForwardIterator::operator*);

  py::class_<EMPlanner2D::BackwardIterator>(m, "EMPlannerBackwardIterator")
      .def_property_readonly("edge", &EMPlanner2D::BackwardIterator::operator*);

  py::class_<EMPlanner2D> planner2d(m, "EMPlanner2D");
      planner2d.def(py::init<const EMPlanner2D::Parameter &,
                    const BearingRangeSensorModel &,
                    const SimpleControlModel &>())
      .def("get_parameter", &EMPlanner2D::getParameter)
      .def("set_parameter", &EMPlanner2D::setParameter)
      .def("iter_solution", [](const EMPlanner2D &p) {
        return py::make_iterator(p.cbeginSolution(), p.cendSolution()); })
      .def("iter_rrt", [](const EMPlanner2D &p) {
        return py::make_iterator(p.cbeginRRT(), p.cendRRT()); })
      .def("iter_dubins_library", [](const EMPlanner2D &p) {
        return py::make_iterator(p.cbeginDubinsLibrary(), p.cendDubinsLibrary()); })
      .def("get_dubins_path", &EMPlanner2D::getDubinsPath)
      .def_static("calculate_utility", &EMPlanner2D::calculateUtility)
      .def("optimize", &EMPlanner2D::optimize)
      .def("optimize2", &EMPlanner2D::optimize2)
      .def("rrt_planner", &EMPlanner2D::rrt_planner)
      .def("line_planner", &EMPlanner2D::line_planner)
      .def("simulations_reward", &EMPlanner2D::simulations_reward);

  py::enum_<EMPlanner2D::OptimizationResult>(planner2d, "OptimizationResult")
      .value("SAMPLING_FAILURE", EMPlanner2D::OptimizationResult::SAMPLING_FAILURE)
      .value("NO_SOLUTION", EMPlanner2D::OptimizationResult::NO_SOLUTION)
      .value("SUCCESS", EMPlanner2D::OptimizationResult::SUCCESS)
      .value("TERMINATION", EMPlanner2D::OptimizationResult::TERMINATION)
      .export_values();

  py::enum_<EMPlanner2D::OptimizationAlgorithm>(planner2d, "OptimizationAlgorithm")
      .value("EM_AOPT", EMPlanner2D::OptimizationAlgorithm::EM_AOPT)
      .value("EM_DOPT", EMPlanner2D::OptimizationAlgorithm::EM_DOPT)
      .value("OG_SHANNON", EMPlanner2D::OptimizationAlgorithm::OG_SHANNON)
      .value("SLAM_OG_SHANNON", EMPlanner2D::OptimizationAlgorithm::SLAM_OG_SHANNON)
      .export_values();
}
