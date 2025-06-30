import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

# Parameters for Cartpole plot
files = [
    'RAdqn_2025_temp_clustering.npy',  # QM-DQN
    'dqn_new.npy'      # DQN
    ,"sfql_new.npy"
]
legend = ["Q-M DQN", "DQN", "SFQL"]
indices = [list(range(19)), list(range(19))]
w3 = 25
length = -230
step = 200

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plot each method
for i, file in enumerate(files):
    data = np.load(file)#[indices[i]]
    r_list = np.mean(data, axis=0).flatten()[:length]
    std_r = np.std(data, axis=0).flatten()[:length]

    smoothed_mean = np.convolve(r_list, np.ones(w3), 'valid') / w3
    smoothed_std = np.convolve(std_r, np.ones(w3), 'valid') / w3
    x_axis = np.arange(smoothed_mean.shape[0]) * step

    ax.plot(x_axis, smoothed_mean, label=legend[i])
    ax.fill_between(x_axis, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.25)

# Axis formatting
ax.set_xlabel("Step", fontsize=16)
ax.set_ylabel("Average Return", fontsize=16)
ax.set_title("Cartpole", fontsize=18)
ax.tick_params(axis='both', labelsize=14)
ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
ax.legend(fontsize=14)
ax.grid(True)

# Final layout
plt.tight_layout()
# plt.savefig("cartpole_plot.png", bbox_inches='tight', dpi=600)
plt.show()
