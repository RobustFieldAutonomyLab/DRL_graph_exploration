import numpy as np
import os
import pickle
import sys
from collections import deque
import gc
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader
import Networks
import envs.exploration_env as robot
from policy_train import DeepQ, A2C
import subprocess

# setup the training model and method
training_method = sys.argv[1]
model_name = sys.argv[2]

# setup local file paths
case_path = training_method + "_" + model_name + "/"
object_path = '../data/training_object_data/' + case_path
# load pickle file
full_file_name = object_path + 'saved_training.pkl'
with open(full_file_name, 'rb') as f:
    dgrl_training = pickle.load(f)

# choose training method
if training_method == "Q":
    # load training model
    policy_model_name = object_path + 'Model_Policy.pt'
    target_model_name = object_path + 'Model_Target.pt'
    check_point_p = torch.load(policy_model_name)
    check_point_t = torch.load(target_model_name)
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
    model.load_state_dict(check_point_p)
    modelt.load_state_dict(check_point_t)
    model.to(device)
    modelt.to(device)
    dgrl_training.running(model, modelt)

elif training_method == "A2C":
    # load training model
    policy_model_name = object_path + 'Model_Policy.pt'
    value_model_name = object_path + 'Model_Value.pt'
    check_point_p = torch.load(policy_model_name)
    check_point_v = torch.load(value_model_name)
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
    modela.load_state_dict(check_point_p)
    modelc.load_state_dict(check_point_v)
    modela.to(device)
    modelc.to(device)
    dgrl_training.running(modela, modelc)

with open(full_file_name, 'wb') as f:
    pickle.dump(dgrl_training, f)
