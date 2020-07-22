from configparser import SafeConfigParser
import math
import numpy as np
from pyss2d import *
from utils import *
import build.planner2d as planner2d


def read_dubins_params(config):
    dubins_params = planner2d.DubinsParameter()
    dubins_params.max_w = config.getfloat('Dubins', 'max_w')
    dubins_params.dw = config.getfloat('Dubins', 'dw')
    dubins_params.min_v = config.getfloat('Dubins', 'min_v')
    dubins_params.max_v = config.getfloat('Dubins', 'max_v')
    dubins_params.dv = config.getfloat('Dubins', 'dv')
    dubins_params.dt = config.getfloat('Dubins', 'dt')
    dubins_params.min_duration = config.getfloat('Dubins', 'min_duration')
    dubins_params.max_duration = config.getfloat('Dubins', 'max_duration')
    dubins_params.tolerance_radius = config.getfloat('Dubins', 'tolerance_radius')
    return dubins_params


def read_planner_params(config):
    planner_params = planner2d.EMPlannerParameter()
    planner_params.verbose = config.getboolean('Planner', 'verbose')
    planner_params.seed = config.getint('Planner', 'seed')
    planner_params.max_edge_length = config.getfloat('Planner', 'max_edge_length')
    planner_params.num_actions = config.getint('Planner', 'num_actions')
    planner_params.max_nodes = config.getfloat('Planner', 'max_nodes')
    planner_params.angle_weight = config.getfloat('Planner', 'angle_weight')
    planner_params.distance_weight0 = config.getfloat('Planner', 'distance_weight0')
    planner_params.distance_weight1 = config.getfloat('Planner', 'distance_weight1')
    planner_params.d_weight = config.getfloat('Planner', 'd_weight')
    planner_params.occupancy_threshold = config.getfloat('Planner', 'occupancy_threshold')
    planner_params.safe_distance = config.getfloat('Planner', 'safe_distance')
    planner_params.reg_out = config.getboolean('Planner', 'reg_out')
    algorithm = config.get('Planner', 'algorithm')
    if algorithm == 'EM_DOPT':
        planner_params.algorithm = planner2d.EMPlanner2D.OptimizationAlgorithm.EM_DOPT
    elif algorithm == 'EM_AOPT':
        planner_params.algorithm = planner2d.EMPlanner2D.OptimizationAlgorithm.EM_AOPT
    elif algorithm == 'OG_SHANNON':
        planner_params.algorithm = planner2d.EMPlanner2D.OptimizationAlgorithm.OG_SHANNON
    elif algorithm == 'SLAM_OG_SHANNON':
        planner_params.algorithm = planner2d.EMPlanner2D.OptimizationAlgorithm.SLAM_OG_SHANNON
        planner_params.alpha = config.getfloat('Planner', 'alpha')
    else:
        print(algorithm)

    planner_params.dubins_control_model_enabled = config.getboolean('Planner', 'dubins_control_model_enabled')
    if planner_params.dubins_control_model_enabled:
        planner_params.dubins_parameter = read_dubins_params(config)
    return planner_params


class EMExplorer(SS2D):
    def __init__(self, config_name, verbose=False, save_history=False):
        super(EMExplorer, self).__init__(config_name, verbose)

        self._planner_params = read_planner_params(self._config)
        self._planner = planner2d.EMPlanner2D(self._planner_params, self._sim.sensor_model, self._sim.control_model)

        if self.verbose:
            self._planner_params.pprint()
        self.save_history = save_history

    def calculate_utility(self, distance):
        return planner2d.EMPlanner2D.calculate_utility(self._virtual_map, distance, self._planner_params)

    def plan(self):
        return self._planner.optimize2(self._slam, self._virtual_map) == planner2d.EMPlanner2D.OptimizationResult.SUCCESS

    def rrt_plan(self, goal_key, fron):
        return self._planner.rrt_planner(self._slam, self._virtual_map, goal_key, fron[0], fron[1]) == planner2d.EMPlanner2D.OptimizationResult.SUCCESS

    def line_plan(self, goal_key, fron):
        actions =  self._planner.line_planner(self._slam, self._virtual_map, goal_key, fron[0], fron[1])
        return actions

    def simulations_reward(self, actions):
        return self._planner.simulations_reward(self._slam, self._virtual_map, self._sim, actions)

    def follow_dubins_path(self, steps=3):
        # for edge in self._planner.iter_solution():
        #     plot_virtual_map(edge.second.virtual_map)
        #     plt.savefig('occ{}.png'.format(self.step))
        #     plt.close()
        #     break

        odoms = []
        for edge in self._planner.iter_solution():
            odoms.insert(0, edge.get_odoms())

        for odoms_i in odoms[:steps]:
            for odom in odoms_i[:-1]:
                if self.simulate((odom.x, odom.y, odom.theta), core=False):
                    if self.save_history:
                        self.save()
                    return

            odom = odoms_i[-1]
            self.simulate((odom.x, odom.y, odom.theta), core=True)
            if self.save_history:
                self.save()

    def follow_path(self, steps=3):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        if self._planner_params.dubins_control_model_enabled:
            return self.follow_dubins_path(steps)
        path = []
        for edge in self._planner.iter_solution():
            path.insert(0, edge.get_odoms()[0])
        for odom in path[:steps]:
            if self.simulate((odom.x, odom.y, odom.theta), core=True):
                return True

    def plot(self, path=False):
        if path:
            plot_path(self._planner, None, self._planner_params.dubins_control_model_enabled)
        super(EMExplorer, self).plot()

    def savefig(self, figname=None, path=False):
        if path:
            plot_path(self._planner, None, self._planner_params.dubins_control_model_enabled)
        super(EMExplorer, self).savefig(figname)

    def save(self):
        landmarks = []
        for key, landmark in self._slam.map.iter_landmarks():
            cov = landmark.covariance
            landmarks.append((key, landmark.point.x, landmark.point.y, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]))

        trajectory = []
        for i, pose in enumerate(self._slam.map.iter_trajectory()):
            cov = pose.covariance
            trajectory.append((int(pose.core_vehicle), pose.pose.x, pose.pose.y, pose.pose.theta, cov[0, 0], cov[0, 1],
                               cov[0, 2], cov[1, 0], cov[1, 1], cov[1, 2], cov[2, 0], cov[2, 1], cov[2, 2]))

        ground_truth_landmarks = []
        for key, landmark in self._sim.environment.iter_landmarks():
            ground_truth_landmarks.append((key, landmark.point.x, landmark.point.y))

        ground_truth_trajectory = []
        for i, pose in enumerate(self._sim.environment.iter_trajectory()):
            ground_truth_trajectory.append((pose.pose.x, pose.pose.y, pose.pose.theta))

        virtual_landmarks = []
        for landmark in self._virtual_map.iter_virtual_landmarks():
            cov = landmark.covariance
            virtual_landmarks.append((landmark.probability, cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]))

        np.savez('step{}'.format(self.step),
                 landmarks=np.array(landmarks),
                 trajectory=np.array(trajectory),
                 virtual_landmarks=np.array(virtual_landmarks),
                 ground_truth_landmarks=np.array(ground_truth_landmarks),
                 ground_truth_trajectory=np.array(ground_truth_trajectory)
                 )


def explore(config_file, max_distance=450, verbose=False, save_history=False, save_fig=True):
    explorer = EMExplorer(config_file, verbose, save_history)

    status = 'MAX_DISTANCE'
    planning_count = 0
    planning_time = 0.0
    try:
        for step in range(200):
            if step < 4:
                odom = 0, 0, math.pi / 2.0
                explorer.simulate(odom, core=True)
                if save_fig:
                    explorer.savefig()
            else:
                start = time()
                result = explorer.plan()
                planning_time += time() - start
                planning_count += 1

                if result == planner2d.EMPlanner2D.OptimizationResult.SAMPLING_FAILURE:
                    explorer.simulate((0, 0, math.pi / 4), True)
                elif result == planner2d.EMPlanner2D.OptimizationResult.NO_SOLUTION:
                    status = 'NO SOLUTION'
                    print("no solution")
                    break
                elif result == planner2d.EMPlanner2D.OptimizationResult.TERMINATION:
                    status = 'TERMINATION'
                    print("termination")
                    break
                else:
                    if save_fig:
                        explorer.savefig(path=True)
                    explorer.follow_path(5)
                if save_fig:
                    explorer.savefig(path=False)
                if explorer.distance > max_distance:
                    break
    except Exception as e:
        print(e)

    return status, planning_time / planning_count


if __name__ == '__main__':
    import sys

    config_file = sys.path[0] + '/../envs/exploration_env.ini'
    explore(config_file, 10, False, False, True)
