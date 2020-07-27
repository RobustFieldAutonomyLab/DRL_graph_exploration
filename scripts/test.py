import numpy as np
import time
import gc
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
import Networks
import envs.exploration_env as robot
import matplotlib.pyplot as plt

# define testing parameters
map_size = 40  # 40, 60, 80, 100
TEST = True
PLOT = True  # save testing date if False; only visualize the environment if True

# setup the training model and method
training_method = "A2C"  # DQN, A2C
model_name = "GG-NN"  # GCN, GG-NN, g-U-Net

case_path = training_method + "_" + model_name + "/"
weights_path = "../data/torch_weights/" + case_path
policy_name = training_method + "+" + model_name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if map_size == 40:
    plot_max_step = 400
elif map_size == 60:
    plot_max_step = 1200
elif map_size == 80:
    plot_max_step = 2400
elif map_size == 100:
    plot_max_step = 4500

# choose training method
if training_method == "DQN":
    # load training model
    policy_model_name = weights_path + 'MyModel.pt'
    check_point_p = torch.load(policy_model_name)
    if model_name == "GCN":
        model = Networks.GCN()
    elif model_name == "g-U-Net":
        model = Networks.GraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GG-NN":
        model = Networks.GGNN()
    model.load_state_dict(check_point_p)
    model.to(device)
elif training_method == "A2C":
    # load training model
    policy_model_name = weights_path + 'MyModel.pt'
    check_point_p = torch.load(policy_model_name)
    if model_name == "GCN":
        model = Networks.PolicyGCN()
    elif model_name == "g-U-Net":
        model = Networks.PolicyGraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GG-NN":
        model = Networks.PolicyGGNN()
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
    print("This is example ", str(lo))
    data_all = pd.DataFrame()

    # //////////////////////////////////////////////////////////////////////////////
    # driving by policy
    env = robot.ExplorationEnv(map_size, lo, TEST)
    mode = 'human'

    if PLOT:
        f1 = plt.figure(1)
        f1.set_size_inches((1 + (map_size - 40) / 40) * 6.4, (1 + (map_size - 40) / 40) * 4.8)
        # plot environment
        env.render(mode=mode)

    done = False
    step_t = 0

    while not done:
        # get the input data (X, A)
        adjacency, featrues, globals_features, fro_size = env.graph_matrix()
        node_size = adjacency.shape[0]
        key_size = node_size - fro_size
        s_t, b_t = data_process([adjacency, featrues], device)
        mask = np.zeros([node_size])
        mask[-fro_size:] = 1

        # get the output reward (Y)
        all_actions = env.actions_all_goals()

        # choose an action
        gcn_tt0 = time.time()
        if training_method == "DQN":
            readout_t = test_q(s_t, 0, device, model).cpu().detach().numpy()
            action_index = np.argmax(readout_t[-fro_size:])
        elif training_method == "A2C":
            readout_t = test_a2c(s_t, b_t, mask, device, model).view(-1).cpu().detach().numpy()
            action_index = np.argmax(readout_t)
        gcn_tt1 = time.time()

        # choose an action
        actions = all_actions[key_size + action_index]

        policy_time = gcn_tt1 - gcn_tt0

        if not PLOT:
            data_all = data_all.append({"Computation time": policy_time, "Category": policy_name, "Map size": map_size},
                                       ignore_index=True)

        # move to next view point
        for act in actions:
            obs, done, _ = env.step(act)

            if PLOT:
                # plot environment
                env.render(mode=mode, action_index=action_index)

            step_t += 1
            l_error = env.get_landmark_error()
            entro = map_entropy(obs)
            max_traj_error = env.max_uncertainty_of_trajectory()
            if not PLOT:
                data_all = data_all.append(
                    {"Step": step_t, "Category": policy_name, "Map entropy": entro, "Landmarks error": l_error,
                     "Max localization uncertainty": max_traj_error}, ignore_index=True)
            print('step: ', step_t, 'action: ', act, 'done: ', done, 'explored: ', env.status())

            if done:
                while step_t < plot_max_step:
                    step_t = step_t + 1
                    if not PLOT:
                        data_all = data_all.append(
                            {"Step": step_t, "Category": policy_name, "Map entropy": entro, "Landmarks error": l_error,
                             "Max localization uncertainty": max_traj_error}, ignore_index=True)
                break

        if done:
            del env
            gc.collect()

    return data_all


def data_process(data, device):
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
    batch = torch.zeros(np.shape(s_a)[0], dtype=int).to(device)
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
    if not PLOT:
        data_output.to_csv("../data/test_result/" + str(map_size) + "_" + training_method + "_" + model_name + ".csv",
                           index=False)
