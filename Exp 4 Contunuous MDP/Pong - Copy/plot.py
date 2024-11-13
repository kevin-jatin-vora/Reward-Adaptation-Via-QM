import numpy as np
from matplotlib import pyplot as plt
# runs=3
indices1 = np.arange(10)
indices2 = indices1#[0,1,4]
ours = np.load("dqn_RA.npy")[indices1]
length = -1#1000
w3=150
r_list = np.mean(np.array(ours), axis=0).flatten()[:length]
std_r = np.std(np.array(ours), axis=0).flatten()[:length]
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
plt.plot(r_list2[:], label="Ours")
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)

ql = np.load("dqn.npy")[indices2]
r_list = np.mean(np.array(ql), axis=0).flatten()[:length]
std_r = np.std(np.array(ql), axis=0).flatten()[:length]
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
plt.plot(r_list2[:],label="DQN")
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


ql = np.load("sfql//dqn_sfql.npy")[indices2]
r_list = np.mean(np.array(ql), axis=0).flatten()[:length]
std_r = np.std(np.array(ql), axis=0).flatten()[:length]
std_rewards = np.convolve(std_r, np.ones(w3), 'valid') / w3
r_list2 = np.convolve(r_list, np.ones(w3), 'valid') / w3
plt.plot(r_list2[:],label="SF-DQN")
plt.fill_between(range(r_list2.shape[0]), r_list2 - std_rewards, r_list2 + std_rewards, alpha=0.25)


plt.legend(loc='lower right')