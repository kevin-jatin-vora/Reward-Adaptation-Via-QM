import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from matplotlib.image import imread

# Load .npy files
qm_sqfl_sqb = np.load(r"C:\Users\kevin\OneDrive\Desktop\TMLR 2025\Exp 1 Fixed MDP FIxed R\highway-city-world\memorized rewards\QM+SFQL\QM+SFQL+SQB_ours_MR_learning.npy")
qm_sqfl = np.load(r"C:\Users\kevin\OneDrive\Desktop\TMLR 2025\Exp 1 Fixed MDP FIxed R\highway-city-world\memorized rewards\test_combined.npy")
sqfl = np.load("SFQL_og.npy")
ql = np.load("QL_traditonal.npy")
qm = np.load(r"C:\Users\kevin\OneDrive\Desktop\TMLR 2025\Exp 1 Fixed MDP FIxed R\highway-city-world\memorized rewards\QM+SFQL\QM+SFQL_ours_MR_learning.npy")
sqb = np.load("sqb.npy")

# Plot config
window_size = 200
# length = 2700
sns.set_style("darkgrid")

# Colors
color_map = {
    'SQB': 'red',
    'QM': plt.cm.tab10(0),
    'SFQL': plt.cm.tab10(2),
    'QL': plt.cm.tab10(1),
    'QM+SFQL': 'black',
    'QM+SFQL+SQB': 'maroon'
}

# Confidence interval calculator
def confidence_interval(data, confidence=0.95):
    n = data.shape[0]
    if n < 2:
        return np.zeros(data.shape[1])
    se = stats.sem(data, axis=0, nan_policy='omit')
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# Plot function
def plot_lines(ax, data_list, label_list, color_map, legend_handles, length):
    for label, data in zip(label_list, data_list):
        d = data[:, :length]
        mean_data = np.mean(d, axis=0)
        ci = confidence_interval(d)
        smoothed_mean = np.convolve(mean_data, np.ones(window_size), 'valid') / window_size
        smoothed_ci = np.convolve(ci, np.ones(window_size), 'valid') / window_size
        x_axis = np.arange(smoothed_mean.shape[0]) * 2
        line, = ax.plot(x_axis, smoothed_mean, label=label, color=color_map[label])
        ax.fill_between(x_axis, smoothed_mean - smoothed_ci, smoothed_mean + smoothed_ci, alpha=0.25, color=color_map[label])
        legend_handles[label] = line
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.tick_params(axis='both', which='major', labelsize=18)

# Create 1 row, 3 columns subplot with spacing adjusted
fig, axs = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1.05, 1.05, 0.9]})
legend_handles = {}

# Subplot 1: Q-M, QL, SFQL
plot_lines(axs[0], [qm, ql, sqfl, sqb], ['QM', 'QL', 'SFQL', 'SQB'], color_map, legend_handles, length=2700)

# Subplot 2: Q-M, Q-M+SFQL, Q-M+SFQL+SQB
plot_lines(axs[1], [qm, qm_sqfl, qm_sqfl_sqb], ['QM', 'QM+SFQL', 'QM+SFQL+SQB'], color_map, legend_handles, length=1600)

# Align y-axis limits between subplot 1 and 2
ymins = [ax.get_ylim()[0] for ax in axs[:2]]
ymaxs = [ax.get_ylim()[1] for ax in axs[:2]]
shared_ymin = min(ymins)
shared_ymax = max(ymaxs)

for ax in axs[:2]:
    ax.set_ylim(shared_ymin, shared_ymax)


# Subplot 3: show image
image_path = "QM highway city.png"
img = imread(image_path)
axs[2].imshow(img)
axs[2].axis('off')

# One shared legend at top
fig.legend(
    handles=[legend_handles[k] for k in legend_handles],
    labels=[k for k in legend_handles],
    loc='upper center',
    bbox_to_anchor=(0.4, 1.25),
    ncol=6,
    fontsize=18
)

# Common axis labels
fig.text(0.4, 0.04, 'Step', ha='center', fontsize=18)
fig.text(0.03, 0.6, 'Average Return', va='center', rotation='vertical', fontsize=18)

plt.tight_layout(rect=[0.04, 0.04, 1, 1.12])
plt.savefig("QM_Highway_City_3Subplots_withImage.png", bbox_inches='tight', dpi=600)
plt.show()
