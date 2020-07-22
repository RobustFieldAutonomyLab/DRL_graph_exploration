#ifndef EM_EXPLORATION_PLANNER2D_H
#define EM_EXPLORATION_PLANNER2D_H

#include "em_exploration/Simulation2D.h"
#include "em_exploration/SLAM2D.h"
#include "em_exploration/VirtualMap.h"
#include "em_exploration/RNG.h"
#include <stack>
#include <algorithm>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>
#include <cmath>

namespace em_exploration {

class EMPlanner2D {
 public:

  enum OptimizationResult {
    SAMPLING_FAILURE, /// fail to generate samples
    NO_SOLUTION,      /// found no path with information gain
    SUCCESS,          /// successful
    TERMINATION,      /// map is fully explored
  };

  enum OptimizationAlgorithm {
    EM_AOPT,
    EM_DOPT,
    OG_SHANNON,
    SLAM_OG_SHANNON,
  };

  struct DubinsParameter {
    DubinsParameter() {}
    double max_w;  /// maximum angular velocity [-max_w, max_w]
    double dw;     /// interval used to discretize angular velocity [-max_w, -max_w + dw, ..., max_w]
    double min_v;  /// minimum x-axis linear velocity
    double max_v;  /// maximum x-axis linear velocity
    double dv;     /// interval used to discretize linear velocity [-min_v, -min_v + dv, ..., max_v]
    double min_duration;  /// minimum duration of a dubins path
    double max_duration;  /// maximum duration of a dubins path
    double dt;            /// interval used to discretize duration [min_duration, min_duration + dt, ..., max_duration]
    double tolerance_radius;  /// radius used to search for nearest dubins path
  };

  struct Dubins {
    double v;   /// linear velocity
    double w;   /// angular velocity
    int num_steps;   /// number of steps in a dubins path (duration / dt)
    gtsam::Pose2 end;   /// terminal pose
  };

  struct Parameter {
    void print() const;

    bool verbose;   /// print debug info
    RNG::SeedType seed;

    double angle_weight;   /// angle weight in distance function
    /// w = w0 - (w0 - w1) * exploration_percentage
    double distance_weight0;   /// weight0 of distance in cost function
    double distance_weight1;   /// weight1 of distance in cost function
    double d_weight;   /// decrease of weight if no solution is found (to perform another optimization)
    double max_nodes;   /// maximum nodes in RRT
    double max_edge_length;   /// maximum extension distance
    int num_actions;
    double occupancy_threshold;   /// occupancy probability to be considered as free
    double safe_distance;   /// safe distance
    double alpha; // for SLAM_OG_SHANNON

    OptimizationAlgorithm algorithm;

    bool dubins_control_model_enabled;
    DubinsParameter dubins_parameter;
    bool reg_out;
  };

  struct Node {
    typedef std::shared_ptr<Node> shared_ptr;
    typedef std::weak_ptr<Node> weak_ptr;

    Node(const VehicleBeliefState &state)
        : key(0), state(state), distance(0), uncertainty(1e10), cost(1e10),
          odometry_factors(),
          measurement_factors(),
          map(nullptr), virtual_map(nullptr),
          parent(Node::weak_ptr()), children(), isam(nullptr) {}

    const Map &getMap() const {
      assert(map);
      return *map;
    }

    const VirtualMap &getVirtualMap() const {
      assert(virtual_map);
      return *virtual_map;
    }

    void print() const {
      std::cout << "Node (" << state.pose.x() << ", " << state.pose.y() << ", " << state.pose.theta() << "), key: "
                << key << ", distance: " << distance << ", uncertainty: " << uncertainty << ", cost: " << cost
                << std::endl;
    }

    unsigned int key;
    VehicleBeliefState state;

    double distance;
    double uncertainty;
    double cost;

    int n_dubins;
    std::vector<Pose2> poses;
    std::shared_ptr<gtsam::ISAM2> isam;
    std::vector<gtsam::NonlinearFactor::shared_ptr> odometry_factors;
    std::vector<gtsam::NonlinearFactor::shared_ptr> measurement_factors;

    std::shared_ptr<Map> map;
    std::shared_ptr<VirtualMap> virtual_map;

    Node::weak_ptr parent;
    std::vector<Node::shared_ptr> children;
  };

  struct Edge {
    Node::shared_ptr first;
    Node::shared_ptr second;

    const Node &getFirst() const {
      return *first;
    }

    const Node &getSecond() const {
      return *second;
    }

    std::vector<Pose2> getOdoms() const {
      std::vector<Pose2> odoms;
      Pose2 origin = first->state.pose;
      for (int i = 0; i < second->poses.size(); ++i) {
        odoms.emplace_back(origin.between(second->poses[i]));
        origin = second->poses[i];
      }
      return odoms;
    }
  };

  class ForwardIterator {
   public:
    ForwardIterator() {}
    ForwardIterator(Node::shared_ptr node) {
      for (auto child : node->children)
        s_.push(child);
    }

    bool operator==(const ForwardIterator &other) const {
      if (s_.size() != other.s_.size())
        return false;
      return (s_.empty() && other.s_.empty()) || (s_.top() == other.s_.top());
    }

    bool operator!=(const ForwardIterator &other) const {
      return !operator==(other);
    }

    ForwardIterator operator++() {
      ForwardIterator temp;
      temp.s_ = s_;
      operator++(0);
      return temp;
    }

    ForwardIterator operator++(int) {
      if (!s_.empty()) {
        Node::shared_ptr node = s_.top();
        s_.pop();
        for (auto child : node->children)
          s_.push(child);
      }
      return *this;
    }

    Edge operator*() const {
      Edge edge;
      edge.first = s_.top()->parent.lock();
      edge.second = s_.top();
      return edge;
    }

   private:
    std::stack<Node::shared_ptr> s_;
  };

  class BackwardIterator {
   public:
    BackwardIterator() : node_(nullptr) {}
    BackwardIterator(Node::shared_ptr node) : node_(node) {
      if (!node_->parent.lock())
        node_ = nullptr;
    }

    bool operator==(const BackwardIterator &other) const { return node_ == other.node_; }

    bool operator!=(const BackwardIterator &other) const { return !operator==(other); }

    BackwardIterator operator++() {
      BackwardIterator temp = *this;
      operator++(0);
      return temp;
    }

    BackwardIterator operator++(int) {
      if (node_) {
        node_ = node_->parent.lock();
        if (!node_->parent.lock())
          node_ = nullptr;
      }
      return *this;
    }

    Edge operator*() const {
      Edge edge;
      edge.first = node_->parent.lock();
      edge.second = node_;
      return edge;
    }

    const Node &first() const {
      return *node_->parent.lock();
    }

    const Node &second() const {
      return *node_;
    }


   private:
    Node::shared_ptr node_;
  };

  EMPlanner2D(const Parameter &parameter,
              const BearingRangeSensorModel &sensor_model,
              const SimpleControlModel &control_model);

  inline const Parameter &getParameter() const { return parameter_; }

  inline void setParameter(const Parameter &parameter) { parameter_ = parameter; }

  BackwardIterator cbeginSolution() const { return BackwardIterator(best_node_); }

  BackwardIterator cendSolution() const { return BackwardIterator(); }

  ForwardIterator cbeginRRT() const { return ForwardIterator(root_); }

  ForwardIterator cendRRT() const { return ForwardIterator(); }

  std::vector<Dubins>::const_iterator cbeginDubinsLibrary() const { return dubins_library_.cbegin(); }

  std::vector<Dubins>::const_iterator cendDubinsLibrary() const { return dubins_library_.cend(); }

  const Dubins &getDubinsPath(int n_dubins) const {
    assert(n_dubins >= 0 && n_dubins < dubins_library_.size());
    return dubins_library_[n_dubins];
  }

  bool isSafe(Node::shared_ptr node) const;

  bool isSafe(Node::shared_ptr node1, Node::shared_ptr node2) const;

  bool isReached(EMPlanner2D::Node::shared_ptr node, EMPlanner2D::Node::shared_ptr goal) const;

  Node::shared_ptr sampleNode() const;

  bool connectNodeDubinsPath(Node::shared_ptr node, Node::shared_ptr parent);

  bool connectNode(Node::shared_ptr node, Node::shared_ptr parent);

  Node::shared_ptr nearestNode(Node::shared_ptr node) const;

  std::vector<Node::shared_ptr> neighborNodes(Node::shared_ptr node, double radius) const;

  double distanceBetweenNodes(Node::shared_ptr node1, Node::shared_ptr node2) const;

  double calculateUncertainty(Node::shared_ptr node) const;

  double calculateUncertainty_EM(Node::shared_ptr node) const;

  double calculateUncertainty_OG_SHANNON(Node::shared_ptr node) const;

  double calculateUncertainty_SLAM_OG_SHANNON(Node::shared_ptr node) const;

  double costFunction(Node::shared_ptr node) const;

  static double calculateUncertainty(const VirtualMap &virtual_map);
  static double calculateUtility(const VirtualMap &virtual_map, double distance, const Parameter &parameter);

  void updateTrajectory_EM(Node::shared_ptr leaf);

  void updateTrajectory_OG_SHANNON(Node::shared_ptr leaf);

  void updateTrajectory_SLAM_OG_SHANNON(Node::shared_ptr leaf);

  void updateNodeOccupancyMap(Node::shared_ptr node);

  void updateNodeInformation(Node::shared_ptr node);

  void updateNodeInformation_EM(Node::shared_ptr node);

  void updateNodeInformation_OG_SHANNON(Node::shared_ptr node);

  void updateNode(Node::shared_ptr node);

  OptimizationResult rrt_planner(const SLAM2D &slam, const VirtualMap &virtual_map, int n_key, double fron_0, double fron_1);

  std::vector<Pose2> line_planner(const SLAM2D &slam, const VirtualMap &virtual_map, int n_key, double fron_0, double fron_1);

  double simulations_reward(const SLAM2D &slam,const VirtualMap &virtual_map,const Simulator2D &sim, std::vector<Pose2> actions);

  OptimizationResult optimize(const SLAM2D &slam, const VirtualMap &virtual_map);

  OptimizationResult optimize2(const SLAM2D &slam, const VirtualMap &virtual_map);

  Eigen::Vector2d frontier;

 private:

  /// Initialize dubins path library which is used to search for path
  /// with terminal pose close to a desired pose.
  void initializeDubinsPathLibrary();

  /// Initialize with planner with SLAM2D and a virtual map.
  void initialize(const SLAM2D &slam, const VirtualMap &virtual_map);

  Parameter parameter_;
  mutable QRNG qrng_;
  mutable RNG rng_;

  const BearingRangeSensorModel &sensor_model_;
  const SimpleControlModel &control_model_;

  const Map *map_;
  gtsam::Values::shared_ptr values_;
  const VirtualMap *virtual_map_;
  const SLAM2D *slam_;
  gtsam::KeySet updated_keys_;

  Node::shared_ptr root_;

  std::vector<Node::shared_ptr> nodes_;

  Node::shared_ptr best_node_;
  Node::shared_ptr goal_node_;

  int max_nodes_;
  double distance_weight_;
  bool update_distance_weight_;

  std::vector<Dubins> dubins_library_;

  /// SLAM_OG_SHANNON
  double w1_;
  double w2_;
};
}
#endif //EM_EXPLORATION_PLANNER2D_H
