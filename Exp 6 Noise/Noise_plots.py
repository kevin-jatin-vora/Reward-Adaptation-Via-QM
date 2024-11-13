# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:53:34 2024

@author: trive
"""

import math
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# Define base directories for each row of the 3x3 grid
base_dirs = 'autogenerated variant2'  # Path to first folder

# List of files for each subplot in a 3x3 grid
subplot_files = [
    ['ours_0_0.csv', 'QL_0_0.csv', 'SFQL_0_0.csv'],
    ['ours_0_0.25.csv', 'QL_0_0.25.csv', 'SFQL_0_0.25.csv'],
    ['ours_0_0.5.csv', 'QL_0_0.5.csv', 'SFQL_0_0.5.csv']
]

# Define subplot parameters for each domain and method (3 domains, each with 3 different sets of params)
subplot_params = [
    {"w3": 300, "length": 2000}, 
    {"w3": 300, "length": 5000}, 
    {"w3": 300, "length": 6200}
]

# Function to plot data for a single subplot
def plot_subplot(ax, files, base_dir, params, color_map, rid):
    global_min, global_max = float('inf'), float('-inf')
    for i, file in enumerate(files):
        file_path = os.path.join(base_dir, file)  # Get full path
        data = pd.read_csv(file_path).drop("Unnamed: 0", axis=1)
        r_list = np.mean(np.array(data), axis=0).flatten()[:params["length"]]
        std_r = np.std(np.array(data), axis=0).flatten()[:params["length"]]
        std_rewards = np.convolve(std_r, np.ones(params["w3"]), 'valid') / params["w3"]
        r_list2 = np.convolve(r_list, np.ones(params["w3"]), 'valid') / params["w3"]
        
        label = file.split('_')[0]
        if label not in color_map:
            color_map[label] = plt.cm.tab10(len(color_map))
        
        ax.plot(r_list2, label=label, color=color_map[label])
        ax.fill_between(
            range(r_list2.shape[0]), 
            r_list2 - std_rewards, 
            r_list2 + std_rewards, 
            alpha=0.25, 
            color=color_map[label]
        )
        ax.set_title("R = +/-" + file.split("_")[-1].replace(".csv", ""))

        # Set axis label font size and reduce ticks for readability
        ax.tick_params(axis='both', labelsize=14)  # Increase tick font size
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Reduce x-axis ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # Reduce y-axis ticks

        # Update global min and max
        global_min = min(global_min, np.min(r_list2 - std_rewards))
        global_max = max(global_max, np.max(r_list2 + std_rewards))
    
    return global_min, global_max

# Create a 3x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
color_map = {}

# Store row-wise min and max values for common y-axis limits
row_mins, row_maxs = [], []
for row in range(3):  # Rows represent the different subplots within each domain
    global_min, global_max = plot_subplot(
        axs[row],  # Domain row-wise
        subplot_files[row], 
        base_dirs, 
        subplot_params[row],  # Access the correct parameters for each row
        color_map,
        row
    )
    row_mins.append(global_min)
    row_maxs.append(global_max)

# Set common y-axis limits for the entire row
common_ymin, common_ymax = min(row_mins), max(row_maxs)
for row in range(3):
    axs[row].set_ylim(common_ymin, common_ymax)

# Set common x and y labels for the entire plot
fig.text(0.5, 0.005, 'Episode', ha='center', va='center', fontsize=14)
fig.text(0.005, 0.5, 'Average Reward', ha='center', va='center', rotation='vertical', fontsize=14)

# Add a common legend outside the subplots, but closer to the figures
handles, labels = [], []
for label, color in color_map.items():
    handles.append(plt.Line2D([0], [0], linestyle='-', color=color))
    labels = ["Q-M", "QL", "SFQL"]

plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, labelspacing=0.1, fontsize=16)

# Apply tight layout to avoid overlapping labels
plt.tight_layout(rect=[0, 0, 1, 0.95])
# Show the plot
plt.savefig('approx.jpeg', bbox_inches='tight', dpi=600)
# Show the plot
plt.show()
