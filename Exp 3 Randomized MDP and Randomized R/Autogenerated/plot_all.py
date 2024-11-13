import os
import re
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")
subplots_files =[]
for n in [1,3,5]:
    subplots_files.append([f'ours_0_{n}.csv',f'QL_0_{n}.csv',f'SFQL_0_{n}.csv'])


# Define subplot parameters
subplot_params = [
    {
     "w3": 300,
     "length": 4000},
    {
     "w3": 300,
     "length": 4000},
    {
     "w3": 300,
     "length": 4000},
    {
     "w3": 150,
     "length": 550,}
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
        # tmp = f.split('_')
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
        
import math

# Determine number of columns dynamically
num_cols = 3

# Create subplots dynamically with up to 3 subplots in each row
num_rows = 1
fig, axs = plt.subplots(math.ceil(num_rows / num_cols), num_cols, figsize=(20, 5))
# Track colors assigned to each label
color_map = {}

# Plot data for each subplot
for i, ax in enumerate(axs):
    plot_subplot(ax, subplots_files[i], subplot_params[i], color_map)
    ax.set_title("ASBF " +subplots_files[i][0].split('_')[-1].replace('.csv',''))
    
# Set common title
#plt.suptitle('Average Reward per Episode over 50 Runs', fontsize=16)

# Add legend
handles, labels = [], []
for label, color in color_map.items():
    handles.append(plt.Line2D([0], [0], linestyle='-', color=color))
    labels.append(label)
plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, labelspacing=0.)
# Set common x and y labels for the entire plot
fig.text(0.5, 0.009, 'Episode', ha='center', va='center')
fig.text(0.005, 0.5, 'Average Reward', ha='center', va='center', rotation='vertical')
plt.rcParams['font.size'] = '10'

plt.tight_layout()
plt.savefig('linear_1.png', bbox_inches='tight', dpi=600)

plt.show()


