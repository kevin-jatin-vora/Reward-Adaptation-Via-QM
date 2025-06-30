import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("darkgrid")

# Define base directories for each game
base_dirs = [ 'Lunar Lander', 'Cartpole', 'Mountain Car']

# File names per method
subplot_files = [
    ['qm.npy', 'dqn.npy', 'sfql.npy', 'sqb.npy'],
    ['qm.npy', 'dqn.npy', 'sfql.npy', 'sqb_b1_init2.npy'],
    ['qm.npy', 'dqn.npy', 'sfql.npy', 'sqb.npy']
]

# Parameters per game
subplot_params = [
    {"w3": 100, "length": -1, "step": 2000},
    {"w3": 10, "length": -1, "step": 200},
    {"w3": 50, "length": -1, "step": 2000},
    
]

# Function to compute confidence interval
def confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return 0
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# Plotting function with CI
def plot_subplot(ax, files, base_dir, params, color_map):
    for i, file in enumerate(files):
        file_path = os.path.join(base_dir, file)
        data = np.load(file_path)
        data = data[:min(len(data),15), :params["length"]]

        mean_rewards = np.mean(data, axis=0)
        ci_values = np.array([confidence_interval(data[:, t]) for t in range(data.shape[1])])

        # Smoothing
        w = params["w3"]
        smooth = lambda x: np.convolve(x, np.ones(w), 'valid') / w

        smoothed_mean = smooth(mean_rewards)
        smoothed_upper = smooth(mean_rewards + ci_values)
        smoothed_lower = smooth(mean_rewards - ci_values)

        x_axis = np.arange(len(smoothed_mean)) * params["step"]

        label = file.replace('sqb_b1_init2','sqb').split('_')[0]
        
        if label not in color_map:
            color_map[label] = plt.cm.tab10(len(color_map))

        ax.plot(x_axis, smoothed_mean, label=label, color=color_map[label])
        ax.fill_between(x_axis, smoothed_lower, smoothed_upper, alpha=0.25, color=color_map[label])
        ax.set_title(base_dir, fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))

# Create a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
color_map = {}

# Plot each subplot
for i in range(3):
    plot_subplot(axs[i], subplot_files[i], base_dirs[i], subplot_params[i], color_map)

# Global labels
fig.text(0.5, 0.005, 'Steps', ha='center', va='center', fontsize=14)
fig.text(0.005, 0.5, 'Average Reward', ha='center', va='center', rotation='vertical', fontsize=14)

# Custom legend
handles = [plt.Line2D([0], [0], linestyle='-', color=color_map[lbl]) for lbl in color_map]
labels = ["Q-M-DQN", "DQN", "SFQL-DQN", "SQB-DQN"]
plt.figlegend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=12)

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('continuous.jpeg', bbox_inches='tight', dpi=600)
plt.show()
