import os
import re
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
subplots_files =[]
for n in ['1', '2', '3', '4']: #['0_1_500', '0_2_500', '0_3_500']
    subplots_files.append([f'ours_{n}.csv',f'QL_{n}.csv',f'sfql_{n}.csv'])

f = open("test_ep.txt", "r")
test_ep = f.read()
f.close()
# Define subplot parameters
subplot_params = [
    {
     "w3": 150,
     "length": 1000},
    {
     "w3": 150,
     "length": 900},
    {
     "w3": 150,
     "length": 1300},
    {
     "w3": 150,
     "length": 1300},
    # Add parameters for other subplots as needed
]

# Function to extract number from file name
def extract_number(file):
    match = re.search(r'\d+', file)
    return int(match.group()) if match else None

# Function to plot data for a single subplot
def plot_subplot(ax, files, params, color_map):
    cnt=0
    for file in files:
        f=file
        data = pd.read_csv(file).drop("Unnamed: 0", axis=1)        
        r_list = np.mean(np.array(data), axis=0).flatten()[:params["length"]]
        tmp = f.split('_')
        # if(tmp[0]=='QL' and tmp[1]=='80'):
        #     r_list+=0.004
        std_r = np.std(np.array(data), axis=0).flatten()[:params["length"]]
        std_rewards = np.convolve(std_r, np.ones(params["w3"]), 'valid') / params["w3"]
        r_list2 = np.convolve(r_list, np.ones(params["w3"]), 'valid') / params["w3"]
        
        label = f.split('_')[0]  # Remove .csv extension for label
        
        if label not in color_map:
            color_map[label] = plt.cm.tab10(len(color_map))
        
        ax.plot(r_list2[:], label=label, color=color_map[label])
        ax.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25, color=color_map[label])
        ax.set_xlabel(f'x {test_ep}', fontsize=10, loc='right')

    # ax.set_xlabel('Episode')
    # ax.set_ylabel('Average Reward')

import math

# Determine number of columns dynamically
num_cols = 4

# Create subplots dynamically with up to 3 subplots in each row
num_rows = 1
fig, axs = plt.subplots(math.ceil(num_rows / num_cols), num_cols, figsize=(20, 5))
# Track colors assigned to each label
color_map = {}

t=[1,2,3,4]
# Plot data for each subplot
for i, ax in enumerate(axs):
    plot_subplot(ax, subplots_files[i], subplot_params[i], color_map)
    ax.set_title("Max SBF = " +str(t[i]), fontsize=16)
    
# Set common title
#plt.suptitle('Average Reward per Episode over 50 Runs', fontsize=16)

# Add legend
handles, labels = [], []
for label, color in color_map.items():
    handles.append(plt.Line2D([0], [0], linestyle='-', color=color))
    #labels.append(label)
labels = ['Q-M', "QL", "SFQL"]
plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, labelspacing=0., fontsize=16)
# Set common x and y labels for the entire plot
fig.text(0.5, 0.009, 'Step', ha='center', va='center', fontsize=16)
fig.text(0.005, 0.5, 'Average Reward', ha='center', va='center', rotation='vertical', fontsize=16)
plt.rcParams['font.size'] = '10'

plt.tight_layout()
plt.savefig('DE.jpeg', bbox_inches='tight', dpi=600)

plt.show()


