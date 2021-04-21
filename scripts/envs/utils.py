import os
import math
import shutil
from configparser import ConfigParser
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import trimboth
import build.ss2d as ss2d

#######################################
import matplotlib

# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Wedge
matplotlib.rcParams['legend.fancybox'] = False
# matplotlib.rcParams['legend.framealpha'] = 1.0
matplotlib.rcParams['legend.edgecolor'] = 'k'
#######################################

#######################################
from functools import wraps
from time import time


def timeit(func):
    @wraps(func)
    def wrap(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        print('func: %r took: %2.4f sec' % (func.__name__, te - ts))
        return result

    return wrap


#######################################

def load_config(config_name):
    config = ConfigParser(inline_comment_prefixes=';')
    config.read(config_name)
    return config

#######################################
def plot_cov_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_info_ellipse(pos, info, nstd=2, ax=None, **kwargs):
    def eigsorted(info):
        vals, vecs = np.linalg.eigh(info)
        vals = 1.0 / vals
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(info)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_environment(env, ax=None, trajectory=True, label=False):
    if ax is None:
        ax = plt.gca()

    for key, landmark in env.iter_landmarks():
        ax.plot(landmark.point.x, landmark.point.y, 'k+')
        if label:
            ax.text(landmark.point.x, landmark.point.y, str(int(key)),
                    size='smaller', color='k')

    if trajectory:
        x, y = [], []
        for pose in env.iter_trajectory():
            x.append(pose.pose.x)
            y.append(pose.pose.y)
        ax.plot(x, y, 'k-')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim([env.parameter.min_x, env.parameter.max_x])
    ax.set_ylim([env.parameter.min_y, env.parameter.max_y])
    ax.set_aspect('equal', adjustable='box')


def plot_map(m, ax=None, trajectory=True, label=False, cov=True):
    if ax is None:
        ax = plt.gca()

    for key, landmark in m.iter_landmarks():
        ax.plot(landmark.point.x, landmark.point.y, '+', color='orange')
        if cov:
            plot_info_ellipse([landmark.point.x, landmark.point.y], landmark.information, ec='none',
                              color='orange', alpha=0.5)
        if label:
            ax.text(landmark.point.x, landmark.point.y, str(int(key)),
                    size='smaller', color='k')

    if trajectory:
        x, y = [], []
        for pose in m.iter_trajectory():
            x.append(pose.pose.x)
            y.append(pose.pose.y)
            if cov and pose.core_vehicle:
                plot_cov_ellipse([pose.pose.x, pose.pose.y], pose.global_covariance[:2, :2], ec='none',
                                  color='g', alpha=0.5)
        ax.plot(x, y, 'g-')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim([m.parameter.min_x, m.parameter.max_x])
    ax.set_ylim([m.parameter.min_y, m.parameter.max_y])
    ax.set_aspect('equal', adjustable='box')


def plot_virtual_map(virtual_map, map_params, ax=None, virtual_landmarks=True, alpha=0.5):
    if ax is None:
        ax = plt.gca()

    array = virtual_map.to_array()
    ax.imshow(array, origin='lower', alpha=alpha, cmap='bone_r', vmin=0.0, vmax=1.0,
              extent=[map_params.min_x, map_params.max_x,
                      map_params.min_y, map_params.max_y])

    if not virtual_landmarks:
        return
    for vl in virtual_map.iter_virtual_landmarks():
        plot_info_ellipse((vl.point.x, vl.point.y), vl.information, 0.5, None, ec='gray', fill=None)


def plot_virtual_map_cov(cov_array, max_sigma, map_params, ax=None, alpha=1.0):
    if ax is None:
        ax = plt.gca()

    ax.imshow(cov_array[0], origin='lower', alpha=alpha, cmap='bone_r',
              vmin=-0.2*max_sigma, vmax=max_sigma+0.2*max_sigma,
              extent=[map_params.min_x, map_params.max_x,
                      map_params.min_y, map_params.max_y])
    x_res = (map_params.max_x - map_params.min_x) / cov_array[0].shape[1]
    y_res = (map_params.max_y - map_params.min_y) / cov_array[0].shape[0]
    x = np.arange(0, cov_array[0].shape[1]) * x_res + map_params.min_x + 0.5 * x_res
    y = np.arange(0, cov_array[0].shape[0]) * y_res + map_params.min_y + 0.5 * y_res
    XX, YY = np.meshgrid(x, y)
    ax.quiver(XX, YY, x_res * np.cos(cov_array[1]), x_res * np.sin(cov_array[1]),
              headwidth=0, headlength=0, pivot='mid', angles='xy', scale_units='xy', scale=1)


def plot_pose(pose, sensor_params=None, ax=None):
    if ax is None:
        ax = plt.gca()

    length = 0.3
    if sensor_params:
        min_bearing = ss2d.Rot2(sensor_params.min_bearing + pose.theta).theta
        max_bearing = ss2d.Rot2(sensor_params.max_bearing + pose.theta).theta
        ax.plot([pose.x, pose.x + sensor_params.max_range * math.cos(pose.theta)],
                [pose.y, pose.y + sensor_params.max_range * math.sin(pose.theta)],
                color='deepskyblue', alpha=0.3)
        ax.plot([pose.x, pose.x + sensor_params.max_range * math.cos(min_bearing)],
                [pose.y, pose.y + sensor_params.max_range * math.sin(min_bearing)],
                color='deepskyblue', alpha=0.3)
        ax.plot([pose.x, pose.x + sensor_params.max_range * math.cos(max_bearing)],
                [pose.y, pose.y + sensor_params.max_range * math.sin(max_bearing)],
                color='deepskyblue', alpha=0.3)
        fov = Wedge((pose.x, pose.y), sensor_params.max_range,
                    math.degrees(min_bearing), math.degrees(max_bearing),
                    width=(sensor_params.max_range - sensor_params.min_range),
                    color="deepskyblue", ec='none', alpha=0.3)
        ax.add_artist(fov)
    else:
        ax.arrow(pose.x, pose.y, length * math.cos(pose.theta), length * math.sin(pose.theta),
                 head_width=0.2, head_length=0.4, fc='k', ec='k')


def plot_measurements(origin, meas, ax=None, label=False):
    if ax is None:
        ax = plt.gca()

    for key, mea in meas:
        point = mea.transform_from(origin)
        ax.plot(point.x, point.y, 'o',
                color='royalblue', mfc='none', alpha=0.5)
        if label:
            ax.text(point.x, point.y, str(int(key)),
                    size='smaller', color='royalblue', alpha=0.5)


def plot_path(planner, ax=None, dubins=False, cov=True, rrt=True):
    if ax is None:
        ax = plt.gca()

    if dubins:
        max_cost = 0
        min_cost = 1e10
        cmap = matplotlib.cm.get_cmap('viridis')
        for edge in planner.iter_rrt():
            if edge.second.cost < min_cost:
                min_cost = edge.second.cost
            if edge.second.cost > max_cost:
                max_cost = edge.second.cost

        if rrt:
            for edge in planner.iter_rrt():
                x = [p.x for p in edge.second.poses]
                y = [p.y for p in edge.second.poses]
                # color = cmap((edge.second.cost - min_cost) / (max_cost - min_cost))
                color = 'purple'
                ax.plot(x, y, '-', color=color, alpha=0.3, linewidth=1.0)

                if cov and edge.second.cost < 1e9:
                    plot_cov_ellipse([x[-1], y[-1]], edge.second.state.global_covariance[:2, :2], ec='none',
                                      color='red', alpha=0.5)

        for edge in planner.iter_solution():
            x = [p.x for p in edge.second.poses]
            y = [p.y for p in edge.second.poses]
            ax.plot(x, y, '-', color='darkred', linewidth=1.0)

    else:
        for edge in planner.iter_rrt():
            x = [edge.first.state.pose.x, edge.second.state.pose.x]
            y = [edge.first.state.pose.y, edge.second.state.pose.y]
            ax.plot(x, y, '-', color='orchid', alpha=0.5, mew=0.3)

        for edge in planner.iter_solution():
            x = [edge.first.state.pose.x, edge.second.state.pose.x]
            y = [edge.first.state.pose.y, edge.second.state.pose.y]
            ax.plot(x, y, '-', color='purple', mew=0.3, alpha=0.5)


def plot_dubins_library(planner):
    x, y = [], []
    for dubins in planner.iter_dubins_library():
        x.append(dubins.end.x)
        y.append(dubins.end.y)
    plt.plot(x, y, '.', alpha=0.5)
    plt.axis('equal')
    plt.savefig('dubins_library.png', dpi=200, bbox='tight')


def plot_samples(virtual_map):
    for i in range(virtual_map.get_sampled_map_size()):
        m = virtual_map.get_sampled_map(i)

        for key, landmark in m.iter_landmarks():
            plt.plot(landmark.point.x, landmark.point.y, '+', color='orange')

            x, y = [], []
            for pose in m.iter_trajectory():
                x.append(pose.pose.x)
                y.append(pose.pose.y)
            plt.plot(x, y, 'g-')


#######################################


def measure_distance(pose1, pose2, angle_weight=0.5):
    angle = pose1[2] - pose2[2]
    angle = math.atan2(math.sin(angle), math.cos(angle))
    return math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2 + (angle * angle_weight) ** 2)


def measure_uncertainty(cov, trace):
    if cov.ndim != 2:
        n = int(math.sqrt(cov.size))
        cov = cov.reshape((n, n))

    if trace:
        return cov.trace()
    else:
        return np.linalg.det(cov)


def measure_entropy(virtual_landmarks):
    e = 0.0
    for vl in virtual_landmarks:
        e += -vl[0] * math.log(vl[0]) - (1 - vl[0]) * math.log(1 - vl[0])
    return e


def get_landmarks_error(folder):
    os.chdir(folder)
    for step in range(1000, 0, -1):
        try:
            landmarks = np.atleast_2d(np.loadtxt('landmarks{}.csv'.format(step)))
            ground_truth_landmarks = np.atleast_2d(np.loadtxt('ground_truth_landmarks{}.csv'.format(step)))
        except IOError:
            continue

        ground_truth_landmarks = {int(round(l[0])): l[1:3] for l in ground_truth_landmarks}
        landmarks = {int(round(l[0])): l[1:3] for l in landmarks}

        error = 0.0
        num = 0
        for key, value in landmarks.iteritems():
            if key in ground_truth_landmarks:
                error += np.linalg.norm(value - ground_truth_landmarks[key])
                num += 1

        os.chdir('..')
        return folder, error / num


def get_trajectory_uncertainty(folder, trace, fixed_distances):
    os.chdir(folder)

    distances = []
    uncertainties = []
    for step in range(1, 1000):
        try:
            data = np.load('step{}.npz'.format(step))
            trajectory = np.atleast_2d(data['trajectory'])
        except IOError:
            continue

        distance = sum(measure_distance(pose1, pose2)
                       for pose1, pose2 in zip(trajectory[:, 1:4], trajectory[1:, 1:4]))
        distances.append(distance)

        trajectory = trajectory[trajectory[:, 0] == 1, :]
        max_uncertainties = [measure_uncertainty(cov, trace) for cov in trajectory[:, 4:]]
        uncertainties.append(max(max_uncertainties))

    if len(distances) == 0:
        print('Empty', folder)

    os.chdir('..')
    if distances[0] > fixed_distances[0]:
        distances.insert(0, fixed_distances[0] - 1)
        uncertainties.insert(0, uncertainties[0])
    if distances[-1] < fixed_distances[-1]:
        distances.append(fixed_distances[-1] + 1)
        uncertainties.append(uncertainties[-1])
    f = interp1d(distances, uncertainties)
    return folder, f(fixed_distances)


def get_map_entropy(folder, fixed_distances):
    os.chdir(folder)

    distances = []
    entropy = []
    for step in range(1, 1000):
        try:
            data = np.load('step{}.npz'.format(step))
            virtual_landmarks = np.atleast_2d(data['virtual_landmarks'])
            trajectory = np.atleast_2d(data['trajectory'])
        except IOError:
            continue

        distance = sum(measure_distance(pose1, pose2)
                       for pose1, pose2 in zip(trajectory[:, 1:4], trajectory[1:, 1:4]))
        distances.append(distance)

        entropy.append(measure_entropy(virtual_landmarks) / len(virtual_landmarks))

    if len(distances) == 0:
        print('Empty', folder)

    os.chdir('..')
    if distances[0] > fixed_distances[0]:
        distances.insert(0, fixed_distances[0] - 1)
        entropy.insert(0, entropy[0])
    if distances[-1] < fixed_distances[-1]:
        distances.append(fixed_distances[-1] + 1)
        entropy.append(entropy[-1])
    f = interp1d(distances, entropy)
    return folder, f(fixed_distances)


def get_landmarks_uncertainty(folder, trace, num_landmarks, uncertainty0, fixed_distances):
    os.chdir(folder)

    distances = []
    uncertainties = []
    for step in range(1, 1000):
        try:
            data = np.load('step{}.npz'.format(step))
            landmarks = np.atleast_2d(data['landmarks'])
            trajectory = np.atleast_2d(data['trajectory'])
        except IOError:
            continue

        distance = sum(measure_distance(pose1, pose2)
                       for pose1, pose2 in zip(trajectory[:, 1:4], trajectory[1:, 1:4]))
        distances.append(distance)

        if landmarks.shape[1] == 0:
            uncertainties.append(num_landmarks * uncertainty0)
        else:
            uncertainties.append(np.sum(measure_uncertainty(landmark[3:], trace) for landmark in landmarks) + \
                                 (num_landmarks - landmarks.shape[0]) * uncertainty0)

    if len(distances) == 0:
        print('Empty', folder)

    os.chdir('..')
    if distances[0] > fixed_distances[0]:
        distances.insert(0, fixed_distances[0] - 1)
        uncertainties.insert(0, uncertainties[0])
    if distances[-1] < fixed_distances[-1]:
        distances.append(fixed_distances[-1] + 1)
        uncertainties.append(uncertainties[-1])
    f = interp1d(distances, uncertainties)
    return folder, f(fixed_distances)


def plot_from_folder(folder):
    try:
        os.chdir(folder)
    except OSError:
        return
    config = ConfigParser()
    config.read(folder + '.ini')
    ext = 5.0
    min_x = config.getfloat('Environment', 'min_x') - ext
    max_x = config.getfloat('Environment', 'max_x') + ext
    min_y = config.getfloat('Environment', 'min_y') - ext
    max_y = config.getfloat('Environment', 'max_y') + ext

    for step in range(1, 1000):
        try:
            data = np.load('step{}.npz'.format(step))
            landmarks = np.atleast_2d(data['landmarks'])
            trajectory = np.atleast_2d(data['trajectory'])
            # ground_truth_landmarks = np.atleast_2d(data['ground_truth_landmarks'])
            # ground_truth_trajectory = np.atleast_2d(data['ground_truth_trajectory'])
        except IOError:
            continue

        plt.plot(landmarks[:, 1], landmarks[:, 2], '+', color='orange')
        # plt.plot(ground_truth_landmarks[:, 1], ground_truth_landmarks[:, 2], '+', color='k')
        plt.plot(trajectory[:, 1], trajectory[:, 2], '-', color='g')
        # plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], '-', color='k')

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        plt.gca().set_aspect('equal', adjustable='box')

        plt.savefig('step{}.png'.format(step), bbox='tight')
        plt.close()

    os.chdir('..')


def get_folders():
    folders = []

    try:
        status = open('status.txt').read()
    except IOError as e:
        print(e)
        return folders

    for dirpath, dirnames, filenames in os.walk('.'):
        folder_name = os.path.split(dirpath)[1]
        if folder_name is not '.' and folder_name in status:
            folders.append(folder_name)
    return folders


def measure_error(results, one_dim=False):
    metrics = {}
    for result in results:
        # option = result[0].split('_', 1)[1]
        option = result[0].rsplit('_', 1)[0]
        if option in metrics:
            metrics[option].append(result[1])
        else:
            metrics[option] = [result[1]]

    errors = {}
    for option, result in metrics.iteritems():
        print(option, len(result))
        if not one_dim:
            result = np.atleast_2d(result)
        else:
            result = np.array(result)
        if result.size == 0:
            continue
        result = np.sort(result, 0)
        result = trimboth(result, 0.05, 0)
        errors[option] = np.mean(result, 0), np.sqrt(np.var(result, 0))

    return errors


def clean_folders():
    try:
        status = open('status.txt').read()
    except IOError:
        return

    for dirpath, dirnames, filenames in os.walk('.'):
        folder_name = os.path.split(dirpath)[1]
        if folder_name is not '.' and folder_name not in status:
            shutil.rmtree(folder_name)
