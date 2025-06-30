import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy import stats
import seaborn as sns
sns.set_style("darkgrid")

# ----------------------
# Load and prepare data for bar plot
# ----------------------
all_files = glob.glob("*.csv")
target_files = [f for f in all_files if re.search(r'_(4|9|14|19|24|29)\.csv$', f)]
df_list = [pd.read_csv(file) for file in target_files]
df = pd.concat(df_list, ignore_index=True)

# Add readable domain label
df['DomainLabel'] = 'R +/- ' + df['Domain'].apply(lambda x: x.split('_')[-1].replace('.csv', ''))
avg_pruned = df.groupby('DomainLabel')['Actions Pruned'].mean().reset_index()
avg_pruned = avg_pruned.sort_values(by='Actions Pruned', ascending=False)

# ----------------------
# Reward plot setup
# ----------------------
reward_files = [f'ours_0_{0}.csv', f'ours_0_{0.015}.csv', f'ours_0_{0.03}.csv']
params = {"w3": 1, "length": 90}
colors = ['#191970', '#6A5ACD', '#4169E1']

# Use first 3 pruned values from avg_pruned for labeling
pruned = avg_pruned['Actions Pruned'].iloc[:3].tolist()

def confidence_interval(data, confidence=0.9):
    n = len(data)
    if n < 2:
        return np.zeros_like(data)
    se = stats.sem(data, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

# ----------------------
# Combined Plot
# ----------------------
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# --- Left: Bar chart ---
bars = axs[0].bar(avg_pruned['DomainLabel'], avg_pruned['Actions Pruned'], color='slateblue', width=0.3)
for bar in bars:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=18)

axs[0].set_xlabel('Noise', fontsize=18)
axs[0].set_ylabel('Average Actions Pruned', fontsize=18)
# axs[0].set_title('Average Actions Pruned')
axs[0].tick_params(axis='x', rotation=0)
axs[0].tick_params(axis='both', labelsize=18)

# --- Right: Reward curves with confidence intervals ---
for idx, file in enumerate(reward_files):
    data = pd.read_csv(file).drop("Unnamed: 0", axis=1)
    rewards_array = np.array(data)[:, :params["length"]]

    mean_rewards = np.mean(rewards_array, axis=0)
    ci = confidence_interval(rewards_array)

    smoothed_mean = np.convolve(mean_rewards, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_upper = np.convolve(mean_rewards + ci, np.ones(params["w3"]), 'valid') / params["w3"]
    smoothed_lower = np.convolve(mean_rewards - ci, np.ones(params["w3"]), 'valid') / params["w3"]

    label = f"Actions pruned = {np.round(pruned[idx], 2)}"
    x_axis = np.arange(smoothed_mean.shape[0]) * 40

    axs[1].plot(x_axis, smoothed_mean, label=label, color=colors[idx])
    axs[1].fill_between(x_axis, smoothed_lower, smoothed_upper, alpha=0.2, color=colors[idx])

axs[1].set_xlabel("Step", fontsize=18)
axs[1].set_ylabel("Average Return", fontsize=18)
# Legend on top, outside the plot
axs[1].legend(loc='lower right', ncol=1, fontsize=18)

axs[1].tick_params(axis='both', labelsize=18)

plt.tight_layout()
plt.savefig("combined_bar_reward_plot.jpg", bbox_inches='tight', dpi=600)
plt.show()
