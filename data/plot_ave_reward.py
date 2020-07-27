import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

directory = "reward_data/"
folder_policy_name = [("DQN_GCN", "DQN+GCN"),
                      ("DQN_GG-NN", "DQN+GG-NN"),
                      ("DQN_g-U-Net", "DQN+g-U-Net"),
                      ("A2C_GCN", "A2C+GCN"),
                      ("A2C_GG-NN", "A2C+GG-NN"),
                      ("A2C_g-U-Net", "A2C+g-U-Net")]

data_all = pd.DataFrame({"Step": [], "Reward": [], "Category": []})

for folder, policy in folder_policy_name:
    print(policy)
    csv_name = directory + folder + "/reward_data.csv"
    # read this csv file
    df = pd.read_csv(csv_name)
    # define the length of the csv
    col_num = df.shape[0]
    # define category
    df['Category'] = policy

    data_all = data_all.append(df, ignore_index=True)

data_all['Reward'] = data_all['Reward'].iloc[::-1].rolling(10000).mean()
indexNames = data_all[data_all['Step'] > 1000000 - 10000].index
data_all.drop(indexNames, inplace=True)
data_all = data_all.rename({'Reward': 'Average reward'}, axis='columns')

flatui = ["#1f78b4", "#fb9a99", "#b2df8a", "#fdbf6f", "#e31a1c", "#a6cee3"]
sns.set(style="darkgrid", color_codes=True, font="sans-serif", font_scale=1.5)
sns.set_palette(sns.color_palette(flatui))

fig = plt.figure(1, figsize=(10.4, 4.8))
sns.lineplot(x='Step', y='Average reward', hue='Category', ci=None, data=data_all)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlim((-50000, 1000000))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
fig.tight_layout(pad=0)
plt.show()
