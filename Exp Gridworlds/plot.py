import math
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("darkgrid")

# Define base directories for each row of the 3x3 grid
base_dirs = [
    'Dollar-Euro',  # Path to first folder
    'Racetrack',   # Path to third folder
    'Frozen lake'
]

# List of files for each subplot in a 3x3 grid
subplot_files = [
    [['ours_0_1.csv', 'QL_0_1.csv', 'SFQL_0_1.csv', 'clipped_QL_0_1.csv'],
     ['ours_0_2.csv', 'QL_0_2.csv', 'SFQL_0_2.csv', 'clipped_QL_0_2.csv'],
     ['ours_0_4.csv', 'QL_0_4.csv', 'SFQL_0_4.csv', 'clipped_QL_0_4.csv']],
    
    [['ours_0_1.csv', 'QL_0_1.csv', 'SFQL_0_1.csv', 'clipped_QL_0_1.csv'],
     ['ours_0_5.csv', 'QL_0_5.csv', 'SFQL_0_5.csv', 'clipped_QL_0_5.csv'],
     ['ours_0_7.csv', 'QL_0_7.csv', 'SFQL_0_7.csv', 'clipped_QL_0_7.csv']],
    
    [['ours_0_1.csv', 'QL_0_1.csv', 'SFQL_0_1.csv', 'clipped_QL_0_1.csv'],
     ['ours_0_2.csv', 'QL_0_2.csv', 'SFQL_0_2.csv', 'clipped_QL_0_2.csv'],
     ['ours_0_4.csv', 'QL_0_4.csv', 'SFQL_0_4.csv', 'clipped_QL_0_4.csv']],
]

# Define subplot parameters for each domain and method (3 domains, each with 3 different sets of params)
subplot_params = [
    [ {"w3": 150, "length": 1000}, {"w3": 150, "length": 1000}, {"w3": 150, "length": 1000}, {"w3": 150, "length": 1000} ],  # Domain 1
    [ {"w3": 150, "length": 1300}, {"w3": 150, "length": 1300}, {"w3": 150, "length": 1300}, {"w3": 150, "length": 1300} ],  # Domain 2
    [ {"w3": 150, "length": 900}, {"w3": 150, "length": 900}, {"w3": 150, "length": 900}, {"w3": 150, "length": 1000} ],  # Domain 3
]

# Confidence interval calculation function
def confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return 0, 0  # Cannot calculate CI with less than 2 data points
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h
# Function to plot data for a single subplot with confidence intervals
def plot_subplot(ax, files, base_dir, params, color_map, rid, cid):
    global_min, global_max = float('inf'), float('-inf')
    if(cid==0):
        step=10
    elif(cid==2):
        step = 8
    else:
        step=4
    global_min, global_max = float('inf'), float('-inf')
    confidence_level = 0.95  # Set your desired confidence level

    for i, file in enumerate(files):
        file_path = os.path.join(base_dir, file)  # Get full path
        data = pd.read_csv(file_path).drop("Unnamed: 0", axis=1)
        all_runs_rewards = np.array(data)[:, :params[rid]["length"]]
        mean_rewards = np.mean(all_runs_rewards, axis=0)
        ci_values = np.array([confidence_interval(run_rewards) for run_rewards in all_runs_rewards.T])

        smoothed_mean_rewards = np.convolve(mean_rewards, np.ones(params[rid]["w3"]), 'valid') / params[rid]["w3"]
        smoothed_ci_upper = np.convolve(mean_rewards + ci_values, np.ones(params[rid]["w3"]), 'valid') / params[rid]["w3"]
        smoothed_ci_lower = np.convolve(mean_rewards - ci_values, np.ones(params[rid]["w3"]), 'valid') / params[rid]["w3"]

        x_axis = np.arange(smoothed_mean_rewards.shape[0]) * step
        label = file.split('_')[0]
        if label not in color_map:
            color_map[label] = plt.cm.tab10(len(color_map))

        ax.plot(x_axis[:smoothed_mean_rewards.shape[0]], smoothed_mean_rewards[:], label=label, color=color_map[label])
        ax.fill_between(
            x_axis[:smoothed_mean_rewards.shape[0]],
            smoothed_ci_lower[:],
            smoothed_ci_upper[:],
            alpha=0.25,
            color=color_map[label]
        )
        ax.set_title("SBF = " + file.split("_")[-1].replace(".csv", ""), fontsize=20)

        # Set axis label font size and reduce ticks for readability
        ax.tick_params(axis='both', labelsize=20)  # Increase tick font size
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Reduce x-axis ticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))  # Reduce y-axis ticks

        # Update global min and max
        global_min = min(global_min, np.min(smoothed_ci_lower))
        global_max = max(global_max, np.max(smoothed_ci_upper))

    return global_min, global_max
# Create a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(20, 15))
color_map = {}

buffer=[0.05, 0.6, 0.05]
# Transpose the grid structure to ensure each domain is in one column
for col in range(3):  # Columns represent different domains
    row_mins, row_maxs = [], []
    for row in range(3):  # Rows represent the different subplots within each domain
        global_min, global_max = plot_subplot(
            axs[col][row],  #Domain row wise
            subplot_files[col][row], 
            base_dirs[col], 
            subplot_params[col], 
            color_map,
            row,
            col
        )
        row_mins.append(global_min)
        row_maxs.append(global_max)
        
    # Set common y-axis limits for the entire row
    common_ymin, common_ymax = min(row_mins), max(row_maxs)
    for row in range(3):
        axs[col][row].set_ylim(common_ymin, common_ymax+buffer[col])

# Set common x and y labels for the entire plot
fig.text(0.5, 0, 'Step', ha='center', va='center', fontsize=20)
fig.text(0.005, 0.5, 'Average Return', ha='center', va='center', rotation='vertical', fontsize=20)

# Add a common legend outside the subplots, but closer to the figures
handles, labels = [], []
for label, color in color_map.items():
    handles.append(plt.Line2D([0], [0], linestyle='-', color=color))
    # labels.append(label)
labels = ["Q-M" , "QL", "SFQL", "SQB"]

plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, labelspacing=0.1, fontsize=22)

# Apply tight layout to avoid overlapping labels
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('CI_exp1_4.jpeg', bbox_inches='tight', dpi=600)
# Show the plot
plt.show()
