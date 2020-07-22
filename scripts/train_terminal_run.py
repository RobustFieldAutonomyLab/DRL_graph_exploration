import numpy as np
import os
import pickle
from collections import deque
import gc
import random
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
import Networks
import envs.exploration_env as robot
from policy_train import DeepQ, A2C
import subprocess

# setup the training model and method
training_method = "Q"  # Q, A2C

# Q:
# GCN, GatedGCNet, GraphUNet
#
# A2C:
# GCN{PolicyGCN, ValueGCN},
# GatedGCNet{PolicyGatedGCNet, ValueGatedGCNet},
# GraphUNet{PolicyGraphUNet, ValueGraphUNet}
model_name = "GCN"

# setup local file paths
case_path = training_method + "_" + model_name + "/"
object_path = '../data/training_object_data/' + case_path
log_path = "../data/torch_logs/" + case_path
if not os.path.exists(log_path):
    os.makedirs(log_path)
if not os.path.exists(object_path):
    os.makedirs(object_path)

# tensorboard
writer = SummaryWriter(log_dir=log_path)

# choose training method
if training_method == "Q":
    # # create a training object
    dgrl_training = DeepQ(case_path)
    # defind training parameters
    epoch_nums = dgrl_training.EXPLORE/dgrl_training.epoch
    # dump pickle file
    full_file_name = object_path + 'saved_training.pkl'
    with open(full_file_name, 'wb') as f:
        pickle.dump(dgrl_training, f)
    # load Q training model
    policy_model_name = object_path + 'Model_Policy.pt'
    target_model_name = object_path + 'Model_Target.pt'
    device = torch.device('cuda')
    if model_name == "GCN":
        model = Networks.GCN()
        modelt = Networks.GCN()
    elif model_name == "GraphUNet":
        model = Networks.GraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
        modelt = Networks.GraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GatedGCNet":
        model = Networks.GatedGCNet()
        modelt = Networks.GatedGCNet()
    model.to(device)
    modelt.to(device)
    torch.save(model.state_dict(), policy_model_name)
    torch.save(modelt.state_dict(), target_model_name)

elif training_method == "A2C":
    # create a training object
    dgrl_training = A2C(case_path)
    # defind training parameters
    epoch_nums = dgrl_training.EXPLORE/dgrl_training.epoch
    # dump pickle file
    full_file_name = object_path + 'saved_training.pkl'
    with open(full_file_name, 'wb') as f:
        pickle.dump(dgrl_training, f)
    # load training model
    policy_model_name = object_path + 'Model_Policy.pt'
    value_model_name = object_path + 'Model_Value.pt'
    device = torch.device('cuda')
    if model_name == "GCN":
        modela = Networks.PolicyGCN()
        modelc = Networks.ValueGCN()
    elif model_name == "GraphUNet":
        modela = Networks.PolicyGraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
        modelc = Networks.ValueGraphUNet(in_channels=5, hidden_channels=1000, out_channels=1000, depth=3)
    elif model_name == "GatedGCNet":
        modela = Networks.PolicyGatedGCNet()
        modelc = Networks.ValueGatedGCNet()
    modela.to(device)
    modelc.to(device)
    torch.save(modela.state_dict(), policy_model_name)
    torch.save(modelc.state_dict(), value_model_name)


for i in range(int(epoch_nums)):
    cmd = "python3 run_training.py " + training_method + " " + model_name
    subprocess.call(cmd, shell=True)
    temp_reward_data = np.loadtxt(object_path + "temp_reward.csv", delimiter=",")
    temp_loss_data = np.loadtxt(object_path + "temp_loss.csv", delimiter=",")
    for j in range(np.shape(temp_reward_data)[0]):
        step_t = temp_reward_data[j][0]
        reward = temp_reward_data[j][1]
        writer.add_scalar('Train/avg_reward', reward, step_t)
    for j in range(np.shape(temp_loss_data)[0]):
        step_t = temp_loss_data[j][0]
        loss = temp_loss_data[j][1]
        writer.add_scalar('Train/loss', loss, step_t)
