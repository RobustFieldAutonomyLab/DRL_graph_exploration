#ifndef EM_EXPLORATION_OCCUPANCYMAP_H
#define EM_EXPLORATION_OCCUPANCYMAP_H

#include <cmath>
#include "em_exploration/Simulation2D.h"
#include "em_exploration/Utils.h"

namespace em_exploration {

#define PROB2LOGODDS(p) (log(p/(1.0 - p)))
#define LOGODDS2PROB(l) (exp(l) / (1.0 + exp(l)))

const double LOGODDS_FREE = PROB2LOGODDS(0.3);
const double LOGODDS_UNKNOWN = PROB2LOGODDS(0.5);
const double LOGODDS_OCCUPIED = PROB2LOGODDS(0.7);
const double MIN_LOGODDS = PROB2LOGODDS(0.05);
const double MAX_LOGODDS = LOGODDS2PROB(0.95);
const double FREE_THRESH = PROB2LOGODDS(0.5);
const double OCCUPIED_THRESH = PROB2LOGODDS(0.5);

class OccupancyMap {
 public:

  class Parameter : public Map::Parameter {
   public:
    Parameter(const Map::Parameter &parameter)
        : Map::Parameter(parameter) { }

    inline double getResolution() const { return resolution_; }
    inline void setResolution(double value) { resolution_ = value; }

    void print() const;

   private:
    double resolution_;
  };

  OccupancyMap(const Parameter &parameter);

  OccupancyMap(const OccupancyMap &other);

  ~OccupancyMap() {
    delete[] map_;
    map_ = nullptr;
  }

  OccupancyMap& operator=(const OccupancyMap &other);

  inline const Parameter &getParameter() const { return parameter_; }

  int getMapSize() const { return rows_ * cols_; }

  void setProbability(int i, double p) { map_[i] = PROB2LOGODDS(p); }

  double getProbability(int i) const { return LOGODDS2PROB(map_[i]); }

  void update(int row, int col, bool free);

  void update(const VehicleBeliefState &state, const BearingRangeSensorModel &sensor_model);

  void update(const Map &map, const BearingRangeSensorModel &sensor_model);

  Eigen::MatrixXd toArray() const;

 private:
  Parameter parameter_;

  int rows_;
  int cols_;
  double *map_;
};

}
#endif //EM_EXPLORATION_OCCUPANCYMAP_H
