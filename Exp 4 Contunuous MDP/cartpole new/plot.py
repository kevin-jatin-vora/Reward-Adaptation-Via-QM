import numpy as np
from matplotlib import pyplot as plt
# runs=3
indices = list(range(22))
ours = np.load("dqn_RA_new_test2.npy")[indices]
length = -1#1000
w3=50
r_list = np.mean(np.array(ours), axis=0).flatten()[:length]
std_r = np.std(np.array(ours), axis=0).flatten()[:length]
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
plt.plot(r_list2[:], label="Ours")
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

ql = np.load("dqn_new.npy")[indices]
r_list = np.mean(np.array(ql), axis=0).flatten()[:length]
std_r = np.std(np.array(ql), axis=0).flatten()[:length]
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
plt.plot(r_list2[:],label="DQN")
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


sfql = np.load("sfql//sfql_dqn_new.npy")[indices]
r_list = np.mean(np.array(sfql), axis=0).flatten()[:length]
std_r = np.std(np.array(sfql), axis=0).flatten()[:length]
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
plt.plot(r_list2[:],label="SF-DQN")
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

plt.xlabel("x200", loc="right")

plt.grid()
plt.legend(loc='lower right')
plt.show()