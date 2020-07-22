import sys
import numpy as np
import os
import time
import gc
import pickle
import pandas as pd
import multiprocessing
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
import Networks
import envs.exploration_env as robot
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

# defind testing parameters
eps_max = 5000
map_size = 40
TEST = True
PLOT = False
use_landmarks = True
large_map_test = False

if map_size == 40:
    plot_max_step = 400
elif map_size == 60:
    plot_max_step = 1200
elif map_size == 80:
    plot_max_step = 2400
elif map_size == 100:
    plot_max_step = 4500

# setup the training model and method
training_method = "A2C"  # Q, A2C

# Q:
# GCN, GatedGCNet, GraphUNet
#
# A2C:
# GCN, GG-NN, GraphUNet
model_name = "GatedGCNet"

if training_method == "Q":
    pt = "DQN"
else:
    pt = training_method

if model_name == "GatedGCNet":
    pm = "GG-NN"
elif model_name == "GraphUNet":
    pm = "g-U-Net"
else:
    pm = model_name

case_path = training_method + "_" + model_name + "/"
weights_path = "../data/torch_weights/" + case_path
policy_name = pt + "+" + pm
figure_path = '../data/figures/visualization/'+str(map_size)+'_'+policy_name+'/'

# choose training method
if training_method == "Q":
    # load training model
    policy_model_name = weights_path + 'MyModel.pt'
    check_point_p = torch.load(policy_model_name)
    device = torch.device('cuda')
    if model_name == "GCN":
        model = Networks.GCN()
    elif model_name == "GraphUNet":
        model = Networks.GraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GatedGCNet":
        model = Networks.GatedGCNet()
    model.load_state_dict(check_point_p)
    model.to(device)
elif training_method == "A2C":
    # load training model
    policy_model_name = weights_path + 'MyModel.pt'
    check_point_p = torch.load(policy_model_name)
    device = torch.device('cuda')
    if model_name == "GCN":
        model = Networks.PolicyGCN()
    elif model_name == "GraphUNet":
        model = Networks.PolicyGraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GG-NN" or model_name == "GatedGCNet":
        model = Networks.PolicyGatedGCNet()
    model.load_state_dict(check_point_p)
    model.to(device)


def map_entropy(obs):
    entro = 0
    if map_size == 40:
        diff = -(0.5 * np.log(0.5)) * 1200
    elif map_size == 80:
        diff = -(0.5 * np.log(0.5)) * 2000
    elif map_size == 60:
        diff = -(0.5 * np.log(0.5)) * 1600
    elif map_size == 100:
        diff = -(0.5 * np.log(0.5)) * 2400
    for i in range(np.shape(obs)[0]):
        for j in range(np.shape(obs)[1]):
            entro = entro + obs[i][j] * np.log(obs[i][j])
    return -entro - diff


def generator(lo_name):
    lo = lo_name
    print("This is the ", str(lo), " example")
    data_all = pd.DataFrame()

    # //////////////////////////////////////////////////////////////////////////////
    # driving by policy
    env = robot.ExplorationEnv(map_size, lo, TEST)
    mode = 'human'

    if PLOT:
        f1 = plt.figure(1)
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        # plot environment
        env.render(mode=mode)

    done = False
    step_t = 0

    while not done:
        # get the input data (X, A)
        adjacency, featrues, globals_features, fro_size = env.graph_matrix()
        node_size = adjacency.shape[0]
        key_size = node_size - fro_size
        s_t, b_t = data_process([adjacency, featrues])
        mask = np.zeros([node_size])
        mask[-fro_size:] = 1

        # get the output reward (Y)
        all_actions = env.actions_all_goals()

        # choose an action
        gcn_tt0 = time.time()
        if training_method == "Q":
            readout_t = test_q(s_t, 0, device, model).cpu().detach().numpy()
            action_index = np.argmax(readout_t[-fro_size:])
        elif training_method == "A2C":
            readout_t = test_a2c(s_t, b_t, mask, device, model).view(-1).cpu().detach().numpy()
            action_index = np.argmax(readout_t)
        gcn_tt1 = time.time()

        # choose an action
        actions = all_actions[key_size + action_index]

        policy_time = gcn_tt1 - gcn_tt0

        data_all = data_all.append({"Computation time": policy_time, "Category": policy_name, "Map size": map_size},
                                     ignore_index=True)

        # move to next view point
        for act in actions:
            obs, reward, done, _ = env.step(act)

            if PLOT:
                # plot environment
                env.render(mode=mode)
                env.show_frontier(action_index)
                f1.set_size_inches((1+(map_size-40)/40)*6.4, (1+(map_size-40)/40)*4.8)
                f1.savefig(figure_path+str(step_t)+'.png')

            step_t += 1
            l_error = env.get_landmark_error()
            entro = map_entropy(obs)
            max_traj_error = env.max_uncertainty_of_trajectory()
            data_all = data_all.append(
                {"Step": step_t, "Category": policy_name, "Map entropy": entro, "Landmarks error": l_error,
                 "Max localization uncertainty": max_traj_error}, ignore_index=True)
            print('step: ', step_t, 'action: ', act, 'done: ', done, 'explored: ', env.status())

            if done:
                while step_t < plot_max_step:
                    step_t = step_t + 1
                    data_all = data_all.append(
                        {"Step": step_t, "Category": policy_name, "Map entropy": entro, "Landmarks error": l_error,
                         "Max localization uncertainty": max_traj_error}, ignore_index=True)
                break

        if done:
            # error_gcn = env.get_landmark_error()
            # dist_gcn = env.get_dist()
            del env
            gc.collect()
            if PLOT:
                plt.waitforbuttonpress()

    return data_all


def data_process(data):
    s_a, s_x = data
    edge_index = []
    edge_attr = []
    edge_set = set()
    for a_i in range(np.shape(s_a)[0]):
        for a_j in range(np.shape(s_a)[1]):
            if (a_i, a_j) in edge_set or (a_j, a_i) in edge_set \
                    or s_a[a_i][a_j] == 0:
                continue
            edge_index.append([a_i, a_j])
            edge_attr.append(s_a[a_i][a_j])
            if a_i != a_j:
                edge_index.append([a_j, a_i])
                edge_attr.append(s_a[a_j][a_i])
            edge_set.add((a_i, a_j))
            edge_set.add((a_j, a_i))
    edge_index = torch.tensor(np.transpose(edge_index), dtype=torch.long)
    x = torch.tensor(s_x, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    state = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = torch.zeros(np.shape(s_a)[0], dtype=int).cuda()
    return state, batch


def test_a2c(data, batch, mask, device, model):
    model.eval()
    data = data.to(device)
    mask = torch.tensor(mask, dtype=bool).cuda()
    pred = model(data, mask, batch)
    return pred


def test_q(data, prob, device, model):
    model.eval()
    data = data.to(device)
    pred = model(data, prob)
    return pred


if __name__ == "__main__":
    test_case = 50
    if not PLOT:
        data_output = pd.DataFrame()
    for i in range(test_case):
        exp_data = generator(i)
        if not PLOT:
            data_output = pd.concat([data_output, exp_data])
    # output.to_csv("../data/test_result/data_m" + str(map_size) + "_" + case_path + ".csv", index=False)
    if not PLOT:
        data_output.to_csv("../data/test_result/" + str(map_size) + "_" + training_method + "_" + model_name + ".csv", index=False)
