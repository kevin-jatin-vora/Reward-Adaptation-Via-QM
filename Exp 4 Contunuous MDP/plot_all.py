import os
import re
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
#plt.rcParams['font.size'] = '18'
sns.set_style("darkgrid")
subplots_files =[]
for n in ['Cartpole new', 'LunarLander - copy', 'Pong - Copy']:
    subplots_files.append([f'plot data//{n}//dqn_RA.npy',f'plot data//{n}//dqn.npy',f'plot data//{n}//sfql_dqn.npy'])


# Define subplot parameters
subplot_params = [
    {
     "w3": 50,
     "length": -1,
     "indices":[list(range(22)),list(range(22)),list(range(22))],
     "step": 200
     },
    {
     "w3": 30,
     "length": -200,
     "indices":[list(range(10)), list(range(10)),list(range(10))],
     "step": 2000
     },
    {
     "w3": 150,
     "length": -1,
     "indices":[list(range(10)), list(range(10)),list(range(10))],
     "step": 500
     }
    # Add parameters for other subplots as needed
]


# Function to plot data for a single subplot
def plot_subplot(i, ax, files, params, color_map):
    cnt=0
    legend = ["QM-DQN", "DQN", "SF-DQN"]
    for file in files:
        if(i==0):
            
            data = np.load(file)[params["indices"][cnt]]
            r_list = np.mean(np.array(data), axis=0).flatten()[:params["length"]]
            std_r = np.std(np.array(data), axis=0).flatten()[:params["length"]]
            std_rewards = np.convolve(std_r, np.ones(params["w3"]), 'valid') / params["w3"]
            r_list2 = np.convolve(r_list, np.ones(params["w3"]), 'valid') / params["w3"]
            
            x_axis =  np.arange(r_list2.shape[0]) * params["step"]
            
            ax.plot(x_axis[:r_list2.shape[0]], r_list2[:], label=legend[cnt])
            ax.fill_between( x_axis[:r_list2.shape[0]], r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
            cnt+=1
        else:
            data = np.load(file)[params["indices"][cnt]]
            r_list = np.mean(np.array(data), axis=0).flatten()[:params["length"]]
            std_r = np.std(np.array(data), axis=0).flatten()[:params["length"]]
            std_rewards = np.convolve(std_r, np.ones(params["w3"]), 'valid') / params["w3"]
            r_list2 = np.convolve(r_list, np.ones(params["w3"]), 'valid') / params["w3"]
            
            x_axis =  np.arange(r_list2.shape[0]) * params["step"]
            
            ax.plot(x_axis[:r_list2.shape[0]], r_list2[:])
            ax.fill_between( x_axis[:r_list2.shape[0]], r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)
            cnt+=1
        ax.set_title(file.replace('plot data//','').split(' ')[0], fontsize=20) #, fontsize=16
        ax.tick_params(axis='both', labelsize=20)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Set number of ticks (adjust as needed)

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
    plot_subplot(i, ax, subplots_files[i], subplot_params[i], color_map)
    # ax.set_title("ASBF " +subplots_files[i][0].split('_')[-1].replace('.csv',''))
    

plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3, labelspacing=0., fontsize=20)
# Set common x and y labels for the entire plot
fig.text(0.5, 0.05, 'Step', ha='center', va='center', fontsize=20)
fig.text(0.005, 0.5, 'Average Return', ha='center', va='center', rotation='vertical', fontsize=20)
#plt.rcParams['font.size'] = '16'

plt.tight_layout()
plt.savefig('continuous.jpeg', bbox_inches='tight', dpi=600)

plt.show()


