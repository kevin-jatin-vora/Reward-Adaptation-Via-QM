import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math
from scipy import stats  # Required for confidence interval calculation

# Load the .npy files
qm_sqfl_sqb = np.load(r"memorized rewards\QM+SFQL\QM+SFQL+SQB_ours_MR_learning.npy")
qm_sqfl = np.load(r"memorized rewards\QM+SFQL\qm_sqfl.npy")
sqfl = np.load("SFQL_og.npy")
ql = np.load("QL_traditonal.npy")
qm = np.load(r"memorized rewards\QM only\QM_only_MR_learning.npy")
sqb = np.load("sqb.npy")

# Define subplot parameters
subplot_params = [
    {
        "w3": 200,  # window size for smoothing
        "length": 2700  # ensuring the same length for all data
    }
]

# Initialize seaborn style
sns.set_style("darkgrid")

# Create subplots dynamically
num_cols = 1
num_rows = 1
fig, axs = plt.subplots(math.ceil(num_rows / num_cols), num_cols, figsize=(10, 6.5))

# Ensure axs is always a list
if not isinstance(axs, np.ndarray):
    axs = [axs]

# Track colors assigned to each label
color_map = {
    'sqb': 'red',
    'QM': plt.cm.tab10(0),
    'SFQL': plt.cm.tab10(2),
    'QL': plt.cm.tab10(1),
    'QM+SFQL': 'black',
    'QM+SFQL+SQB': 'maroon',
    
}

# Prepare data files as a list
data_files = [
    ('QM+SFQL', qm_sqfl),
    ('QM', qm),
    ('SFQL', sqfl),
    ('QL', ql),
    ('sqb', sqb),
    ('QM+SFQL+SQB', qm_sqfl_sqb)
]

# Confidence interval function
def confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    if n < 2:
        return np.zeros(data.shape[1])
    se = stats.sem(data, axis=0, nan_policy='omit')
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# Function to plot data for a single subplot
def plot_subplot(ax, data_files, params, color_map):
    for label, data in data_files:
        # Clip and process the data
        d = data[:, :params["length"]]
        mean_data = np.mean(d, axis=0)
        ci = confidence_interval(d)

        # Smooth the mean
        smoothed_mean = np.convolve(mean_data, np.ones(params["w3"]), 'valid') / params["w3"]
        # Smooth the confidence interval bounds
        smoothed_ci = np.convolve(ci, np.ones(params["w3"]), 'valid') / params["w3"]

        x_axis = np.arange(smoothed_mean.shape[0]) * 2

        # Plot with confidence interval
        ax.plot(x_axis, smoothed_mean, label=label, color=color_map[label])
        ax.fill_between(x_axis, smoothed_mean - smoothed_ci, smoothed_mean + smoothed_ci,
                        alpha=0.25, color=color_map[label])
        
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

# Plot data for the subplot
for ax in axs:
    plot_subplot(ax, data_files, subplot_params[0], color_map)

# Add legend
plt.figlegend(
    handles=[
        plt.Line2D([0], [0], linestyle='-', color=color_map['QM+SFQL'], label='Q-M+SFQL'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['QM'], label='Q-M'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['SFQL'], label='SFQL'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['QL'], label='QL'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['sqb'], label='SQB'),
        plt.Line2D([0], [0], linestyle='-', color=color_map['QM+SFQL+SQB'], label='Q-M+SFQL+SQB'),
    ],
    labels=['Q-M+SFQL','Q-M', 'SFQL', 'QL','SQB', "Q-M+SFQL+SQB"],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),
    ncol=3,
    labelspacing=0.
)

# Set common x and y labels for the entire plot
fig.text(0.55, 0.014, 'Step', ha='center', va='center')
fig.text(0.014, 0.5, 'Average Return', ha='center', va='center', rotation='vertical')
plt.rcParams['font.size'] = '20'

plt.tight_layout()
plt.savefig('HC_memoriezdR_learning_CI.png', bbox_inches='tight', dpi=600)
plt.show()