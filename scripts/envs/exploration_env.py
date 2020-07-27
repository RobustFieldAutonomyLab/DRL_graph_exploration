import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import gc
import gym
from gym import error, spaces
from gym.utils import seeding
sys.path.append(sys.path[0] + '/../..')
sys.path.append(sys.path[0] + '/..')
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

try:
    from envs.pyplanner2d import EMExplorer
    from envs.utils import load_config, plot_virtual_map, plot_virtual_map_cov, plot_path
except ImportError as e:
    raise error.DependencyNotInstalled('{}. Build em_exploration and export PYTHONPATH=build_dir'.format(e))


class ExplorationEnv(gym.Env):
    metadata = {'render.modes': ['human', 'state'],
                'render.pause': 0.001}

    def __init__(self,
                 map_size,
                 env_index,
                 test):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = dir_path + '/exploration_env.ini'
        self._config = load_config(config)
        self.env_index = env_index
        self.map_size = map_size
        self.dist = 0.0
        self.test = test
        self._one_nearest_frontier = False
        self.nearest_frontier_point = 0
        self.seed()
        self._obs = self.reset()
        self._viewer = None
        self.reg_out = self._sim._planner_params.reg_out
        self._max_steps = self._sim._environment_params.max_steps
        self.max_step = self._max_steps

        num_actions = self._sim._planner_params.num_actions
        self._step_length = self._sim._planner_params.max_edge_length
        self._rotation_set = np.arange(0, np.pi * 2, np.pi * 2 / num_actions) - np.pi
        self._action_set = [np.array([np.cos(t) * self._step_length,
                                      np.sin(t) * self._step_length,
                                      t])
                            for t in self._rotation_set]
        self.action_space = spaces.Discrete(n=num_actions)
        assert (len(self._action_set) == num_actions)
        self._done = False
        self.loop_clo = False
        self._frontier = []
        self._frontier_index = []

        self.map_resolution = self._sim._virtual_map_params.resolution
        rows, cols = self._sim._virtual_map.to_array().shape
        self.leng_i_map = rows
        self.leng_j_map = cols
        self._max_sigma = self._sim._virtual_map.get_parameter().sigma0
        min_x, max_x = self._sim._map_params.min_x, self._sim._map_params.max_x
        min_y, max_y = self._sim._map_params.min_y, self._sim._map_params.max_y
        self._pose = spaces.Box(low=np.array([min_x, min_y, -math.pi]),
                                high=np.array([max_x, max_y, math.pi]), dtype=np.float32)
        self._vm_cov_sigma = spaces.Box(low=0, high=self._max_sigma, dtype=np.float32, shape=(rows, cols))
        self._vm_cov_angle = spaces.Box(low=-math.pi, high=math.pi, dtype=np.float32, shape=(rows, cols))
        self._vm_prob = spaces.Box(low=0.0, high=1.0, shape=(rows, cols), dtype=np.float32)
        self.observation_space = spaces.Tuple([self._pose,
                                               self._vm_prob,
                                               self._vm_cov_sigma,
                                               self._vm_cov_angle])
        self.ext = 20.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _get_obs(self):
        cov_array = self._sim._virtual_map.to_cov_array()
        self._obs_show = np.array(
            [self._sim.vehicle_position.x, self._sim.vehicle_position.y, self._sim.vehicle_position.theta]), \
                         self._sim._virtual_map.to_array(), cov_array[0], cov_array[1]
        self._obs = self._sim._virtual_map.to_array()
        return self._obs

    def _get_utility(self, action=None):
        if action is None:
            distance = 0.0
        else:
            angle_weight = self._config.getfloat('Planner', 'angle_weight')
            distance = math.sqrt(action.x ** 2 + action.y ** 2 + angle_weight * action.theta ** 2)
        return self._sim.calculate_utility(distance)

    def step(self, action):
        if self._sim._planner_params.reg_out:
            action = self._action_set[action]
        u1 = self._get_utility()
        self._sim.simulate([action.x, action.y, action.theta])
        self.dist = self.dist + math.sqrt(action.x ** 2 + action.y ** 2)
        u2 = self._get_utility(action)
        return self._get_obs(), self.done(), {}

    def plan(self):
        if not self._sim.plan():
            self._done = True
            return []

        actions = []
        for edge in self._sim._planner.iter_solution():
            if self._sim._planner_params.reg_out:
                actions.insert(0, (np.abs(np.asarray(self._rotation_set) - edge.get_odoms()[0].theta)).argmin())
            else:
                actions.insert(0, edge.get_odoms()[0].theta)
        return actions

    def rrt_plan(self, goal_key):
        if not self._sim.rrt_plan(goal_key, self._frontier):
            self._done = True
            return []

        actions = []
        for edge in self._sim._planner.iter_solution():
            actions.insert(0, edge.get_odoms()[0])
        return actions

    def line_plan(self, goal_key, fro=[0, 0]):
        actions = self._sim.line_plan(goal_key, fro)
        return actions

    def actions_all_goals(self):
        key_size = self._sim._slam.key_size()
        land_size = self.get_landmark_size()
        fro_size = len(self._frontier)
        all_actions = [[]] * (key_size + fro_size)
        # actions for frontiers
        for i, vi in enumerate(self._frontier):
            all_actions[i + key_size] = self.line_plan(key_size, vi)

        return all_actions

    def rewards_all_goals(self, all_actions):
        key_size = self._sim._slam.key_size()
        land_size = self.get_landmark_size()
        fro_size = len(self._frontier)
        rewards = [np.nan] * (key_size + fro_size)

        # calculating rewards for each actions
        for i, _ in enumerate(self._frontier):
            rewards[i + key_size] = self._sim.simulations_reward(all_actions[i + key_size])
        act_max = np.nanargmax(rewards)
        if self.is_nf(act_max):
            self.loop_clo = False
            rewards = np.interp(rewards, (np.nanmin(rewards), np.nanmax(rewards)), (-1.0, 0.0))
        else:
            self.loop_clo = True
            rewards = np.interp(rewards, (np.nanmin(rewards), np.nanmax(rewards)), (-1.0, 1.0))
        rewards[np.isnan(rewards)] = 0
        return rewards

    def status(self):
        return self._sim._virtual_map.explored()

    def done(self):
        return self._done or self._sim.step > self._max_steps or self.status() > 0.85

    def get_landmark_error(self, sigma0=1.0):
        error = 0.0
        for key, predicted in self._sim.map.iter_landmarks():
            landmark = self._sim.environment.get_landmark(key)
            error += np.sqrt((landmark.point.x - predicted.point.x) ** 2 + (landmark.point.y - predicted.point.y) ** 2)
        error += sigma0 * (self._sim.environment.get_landmark_size() - self._sim.map.get_landmark_size())
        return error / self._sim.environment.get_landmark_size()

    def get_dist(self):
        return self.dist

    def get_landmark_size(self):
        return self._sim._slam.map.get_landmark_size()

    def get_key_size(self):
        return self._sim._slam.key_size()

    def print_graph(self):
        self._sim._slam.print_graph()

    def max_uncertainty_of_trajectory(self):
        land_size = self.get_landmark_size()
        self._sim._slam.adjacency_degree_get()
        features = np.array(self._sim._slam.features_out())
        return np.amax(features[land_size:])

    def graph_matrix(self):
        self.frontier()
        trace_map = self._sim._virtual_map.to_cov_trace()
        key_size = self._sim._slam.key_size()
        land_size = self.get_landmark_size()
        fro_size = len(self._frontier)

        self._sim._slam.adjacency_degree_get()
        adjacency = np.array(self._sim._slam.adjacency_out())
        features = np.array(self._sim._slam.features_out())
        adjacency = np.pad(adjacency, ((0, fro_size), (0, fro_size)), 'constant')
        features = np.pad(features, ((0, fro_size), (0, 0)), 'constant')

        robot_location = [self._sim.vehicle_position.x, self._sim.vehicle_position.y]

        # add frontiers to adjacency matrix
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            for j in range(len(self._frontier_index[i])):
                index_node = self._frontier_index[i][j]
                if index_node == 0:
                    self.nearest_frontier_point = i+key_size
                    dist = self.points2dist(frontier_point, robot_location)
                    adjacency[key_size - 1][i + key_size] = dist
                    adjacency[i + key_size][key_size - 1] = dist
                else:
                    dist = self.points2dist(frontier_point, self._sim._slam.get_key_points(index_node - 1))
                    adjacency[index_node - 1][i + key_size] = dist
                    adjacency[i + key_size][index_node - 1] = dist

        # add frontiers to features matrix col 1: trace of cov
        for i in range(fro_size):
            indx = self.coor2index(self._frontier[i][0], self._frontier[i][1])
            f = trace_map[indx[0]][indx[1]]
            features[key_size + i][0] = f

        # add frontiers to features matrix col 2: distance to the robot
        features_2 = np.zeros(np.shape(features))
        for i in range(key_size):
            key_point = self._sim._slam.get_key_points(i)
            dist = self.points2dist(key_point, robot_location)
            features_2[i][0] = dist
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            dist = self.points2dist(frontier_point, robot_location)
            features_2[key_size + i][0] = dist

        # add frontiers to features matrix col 5: direction to the robot
        features_5 = np.zeros(np.shape(features))
        root_theta = self._sim.vehicle_position.theta
        for i in range(key_size):
            key_point = self._sim._slam.get_key_points(i)
            tdiff = self.diff_theta(key_point, robot_location, root_theta)
            features_5[i][0] = tdiff
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            tdiff = self.diff_theta(frontier_point, robot_location, root_theta)
            features_5[key_size + i][0] = tdiff

        # add frontiers to features matrix col 3: probability
        features_3 = np.zeros(np.shape(features))
        for i in range(key_size):
            key_point = self._sim._slam.get_key_points(i)
            indx = self.coor2index(key_point[0], key_point[1])
            probobility = self._obs[indx[0]][indx[1]]
            features_3[i][0] = probobility
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            indx = self.coor2index(frontier_point[0], frontier_point[1])
            probobility = self._obs[indx[0]][indx[1]]
            features_3[key_size + i][0] = probobility

        # add frontiers to features matrix clo 4: index of locations
        features_4 = np.zeros(np.shape(features))
        for i in range(key_size - 1):
            features_4[i][0] = -1
        features_4[key_size - 1][0] = 0
        for i in range(fro_size):
            features_4[key_size + i][0] = 1

        features = np.concatenate((features, features_2, features_5, features_3, features_4), axis=1)

        # create global features
        avg_landmarks_error = np.mean(features[1:land_size + 1][:, 0])
        global_features = np.array([avg_landmarks_error])
        return adjacency, features, global_features, fro_size

    def is_nf(self, id):
        if self.nearest_frontier_point == id:
            return True
        else:
            return False

    def frontier(self):
        vehicle_location = [self._sim.vehicle_position.x, self._sim.vehicle_position.y]

        a = self._obs < 0.45
        free_index_i, free_index_j = np.nonzero(a)

        all_landmarks = []
        all_frontiers = []
        landmark_keys = range(self.get_landmark_size())

        self._frontier = []
        self._frontier_index = []

        for land_key in landmark_keys:
            points = list(self._sim._slam.get_key_points(land_key))
            all_landmarks.append(points)

        for ptr in range(len(free_index_i)):
            cur_i = free_index_i[ptr]
            cur_j = free_index_j[ptr]
            count = 0
            cur_i_min = free_index_i[ptr] - 1 if free_index_i[ptr] - 1 >= 0 else 0
            cur_i_max = free_index_i[ptr] + 1 if free_index_i[ptr] + 1 < self.leng_i_map else self.leng_i_map - 1
            cur_j_min = free_index_j[ptr] - 1 if free_index_j[ptr] - 1 >= 0 else 0
            cur_j_max = free_index_j[ptr] + 1 if free_index_j[ptr] + 1 < self.leng_j_map else self.leng_j_map - 1

            for ne_i in range(cur_i_min, cur_i_max + 1):
                for ne_j in range(cur_j_min, cur_j_max + 1):
                    if 0.49 < self._obs[ne_i][ne_j] < 0.51:
                        count += 1

            if count >= 2:
                ind2co = self.index2coor(cur_i, cur_j)
                if self._sim._map_params.min_x + self.ext <= ind2co[0] <= self._sim._map_params.max_x - self.ext \
                        and self._sim._map_params.min_y + self.ext <= ind2co[
                    1] <= self._sim._map_params.max_y - self.ext:
                    all_frontiers.append(ind2co)

        cur_fro = all_frontiers[self.nearest_frontier(vehicle_location, all_frontiers)]
        self._frontier.append(cur_fro)
        self._frontier_index.append([0])

        if not self._one_nearest_frontier:
            for ip, p in enumerate(all_landmarks):
                cur_fro = all_frontiers[self.nearest_frontier(p, all_frontiers)]
                try:
                    self._frontier_index[self._frontier.index(cur_fro)].append(ip + 1)
                except ValueError:
                    self._frontier.append(cur_fro)
                    self._frontier_index.append([ip + 1])

        not_go = []
        for i, vi in enumerate(self._frontier):
            temp_list = []
            for j, vj in enumerate(self._frontier[i:]):
                if vi == vj and i not in not_go:
                    temp_list.append(i + j)
                    if i + j != i:
                        not_go.append(i + j)
            self._frontier_index.append(temp_list)

    def nearest_frontier(self, point, all_frontiers):
        min_dist = float("Inf")
        min_index = None
        for index, fro_points in enumerate(all_frontiers):
            dist = self.points2dist(point, fro_points)
            if dist < min_dist:
                min_dist = dist
                min_index = index
        return min_index

    def show_frontier(self, index):
        plt.plot(np.transpose(self._frontier)[0], np.transpose(self._frontier)[1], 'mo')
        plt.plot(np.transpose(self._frontier)[0][index], np.transpose(self._frontier)[1][index], 'ro')

    def index2coor(self, matrix_i, matrix_j):
        x = (matrix_j + 0.5) * self.map_resolution + self._sim._map_params.min_x
        y = (matrix_i + 0.5) * self.map_resolution + self._sim._map_params.min_y
        return [x, y]

    def coor2index(self, x, y):
        map_j = int(round((x - self._sim._map_params.min_x) / self.map_resolution - 0.5))
        map_i = int(round((y - self._sim._map_params.min_y) / self.map_resolution - 0.5))
        return [map_i, map_j]

    def points2dist(self, point1, point2):
        dist = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        return dist

    def diff_theta(self, point1, point2, root_theta):
        goal_theta = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
        if goal_theta < 0:
            goal_theta = math.pi * 2 + goal_theta
        if root_theta < 0:
            root_theta = math.pi * 2 + root_theta
        diff = goal_theta - root_theta
        if diff < 0:
            diff = math.pi * 2 + diff
        return diff

    def reset(self):
        self._done = False
        while True:
            # Reset seed in configuration
            if not self.test:
                seed1 = self.np_random.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
                seed2 = self.np_random.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
            else:
                seed1 = self.env_index
                seed2 = self.env_index
            landmark_size = int(self.map_size ** 2 * 0.005)
            self._config.set('Simulator', 'seed', str(seed1))
            self._config.set('Planner', 'seed', str(seed1))
            self._config.set('Simulator', 'lo', str(seed2))
            self._config.set('Environment', 'min_x', str(-self.map_size / 2))
            self._config.set('Environment', 'min_y', str(-self.map_size / 2))
            self._config.set('Environment', 'max_x', str(self.map_size / 2))
            self._config.set('Environment', 'max_y', str(self.map_size / 2))
            self._config.set('Simulator', 'num', str(landmark_size))

            # Initialize new instance and perfrom a 360 degree scan of the surrounding
            self._sim = EMExplorer(self._config)
            for step in range(4):
                odom = 1, 1, math.pi / 2.0
                u1 = self._get_utility()
                self._sim.simulate(odom)

            if self._sim._slam.map.get_landmark_size() < 1:
                print("regenerate a environment")
                self.env_index = self.env_index + 50
                continue

            # Return initial observation
            return self._get_obs()

    def render(self, mode='human', close=False, action_index=-1):
        if close:
            return
        if mode == 'human':
            if self._viewer is None:
                self._sim.plot()
                self._viewer = plt.gcf()
                plt.ion()
                plt.tight_layout()
                plt.xlim((self._sim._map_params.min_x + 14, self._sim._map_params.max_x - 14))
                plt.ylim((self._sim._map_params.min_y + 14, self._sim._map_params.max_y - 14))
                plt.show()
            else:
                self._viewer.clf()
                self._sim.plot()
                # plot_path(self._sim._planner, dubins=False)
                plt.xlim((self._sim._map_params.min_x + 14, self._sim._map_params.max_x - 14))
                plt.ylim((self._sim._map_params.min_y + 14, self._sim._map_params.max_y - 14))
                plt.draw()
            if action_index != -1:
                self.show_frontier(action_index)
            plt.pause(ExplorationEnv.metadata['render.pause'])
        elif mode == 'state':
            # print self._obs
            # assert(len(self._obs_show) == 3)
            print (self._viewer is None)
            if self._viewer is None:
                self._viewer = plt.subplots(1, 3, figsize=(18, 6))
                plot_virtual_map(self._sim._virtual_map, self._sim._map_params,
                                 alpha=1.0, ax=self._viewer[1][0])
                plot_virtual_map_cov(self._obs_show[1:], self._max_sigma,
                                     self._sim._map_params,
                                     alpha=1.0, ax=self._viewer[1][1])
                plt.sca(self._viewer[1][2])
                self._sim.plot()
                plt.ion()
                plt.tight_layout()
                plt.show()
            else:
                self._viewer[1][0].clear()
                plot_virtual_map(self._sim._virtual_map, self._sim._map_params,
                                 alpha=1.0, ax=self._viewer[1][0])
                self._viewer[1][1].clear()
                plot_virtual_map_cov(self._obs_show[1:], self._max_sigma,
                                     self._sim._map_params,
                                     alpha=1.0, ax=self._viewer[1][1])
                self._viewer[1][2].clear()
                plt.sca(self._viewer[1][2])
                self._sim.plot()
                plt.draw()
            plt.pause(ExplorationEnv.metadata['render.pause'])
        else:
            super(ExplorationEnv, self).render(mode=mode)


if __name__ == '__main__':
    import sys

    ExplorationEnv.metadata['render.pause'] = 0.001
    lo_num = 7
    map_size = 40
    total_reward = np.empty([0, 0])
    TEST = False

    mode = 'human'
    env = ExplorationEnv(map_size, lo_num, TEST)
    t = 0
    epoch = 0
    done = False
    flag = False
    actions = []
    env.render(mode=mode)

    for i in range(1000):
        adjacency, featrues, global_features, fro_size = env.graph_matrix()

        key_size = env.get_key_size()
        land_size = env.get_landmark_size()
        all_actions = env.actions_all_goals()
        rewards = env.rewards_all_goals(all_actions)
        # print "rewards: ", rewards, "\n"
        rewards[0:key_size] = np.nan
        act_index = np.nanargmax(rewards)
        max_reward = rewards[act_index]

        actions = all_actions[act_index]
        print("###############################")
        temp_reward = 0
        print("max_reward: ", str(max_reward))

        for a in actions:
            obs, reward, done, _ = env.step(a)
            temp_reward += reward
            env.render(mode=mode, action_index=act_index-key_size)
            print("step: ", str(t), "reward: ", str(reward))

            # print "landmark error: ", env.get_landmark_error()
            t = t + 1
            if done:
                break

        ls = env.get_landmark_size()
        epoch += 1
        adjacency, featrues, global_features, fro_size = env.graph_matrix()
        print('done: ', done, 'explored: ', env.status())

        if done:
            print("done")
            print("total steps: ", t)
            print("epoch is: ", epoch)
            print("error: ", env.get_landmark_error())
            input("Press Enter to continue...")
            del env
            gc.collect()
            env = ExplorationEnv(map_size, lo_num, TEST)
            env.render(mode=mode)
            t = 0
            print("new one")
        flag = False
    plt.pause(1e10)
