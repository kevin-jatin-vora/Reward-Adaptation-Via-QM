import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.chdir(os.getcwd())
S=49
A=4
R1 = np.load("R1.npy")
R2 = np.load("R2.npy")
T = np.load("T.npy")

r=R1+R2
terminal_state = np.load("terminal.npy")
start_state = np.load("initial.npy")[0]
gamma = 0.9

def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(A)
    else:
        return np.argmax(Q[state])


def test_q(e=30):
    global Q
    episode_rewards=[]
    for episode in range(e):
        state = start_state
        total_reward = 0
        step = 0
        while state not in terminal_state and step<max_steps:
            step+=1
            action = epsilon_greedy_policy(state, 0)
            next_state = np.random.choice(S, p=T[state, action, :])
            reward = r[state, action, next_state]
            total_reward += reward
            state = next_state
        episode_rewards.append(total_reward)
    return np.mean(episode_rewards) 

def compute_bounds(Q, dynamics_table, rewards_table, gamma, prior_policy=None):
    S, A, _ = dynamics_table.shape
    beta = 5

    Q_flat = Q.flatten()
    baseline = (np.max(Q_flat) + np.min(Q_flat)) / 2
    Q_flat -= baseline
    Q_max = np.max(Q_flat)
    Q_stable = Q_flat -     
    exp_beta_Q = np.exp(beta * Q_stable)

    transition_dynamics = dynamics_table.reshape(S * A, S).T
    for i in range(transition_dynamics.shape[1]):
        col_sum = np.sum(transition_dynamics[:, i])
        if col_sum > 0:
            transition_dynamics[:, i] /= col_sum
    transition_dynamics_sparse = csr_matrix(transition_dynamics)

    if prior_policy is None:
        prior_policy = np.ones((S, A)) / A

    def pi_from_Q(Q, beta, prior_policy):
        Q_max_local = np.max(Q, axis=1, keepdims=True)
        logits = beta * (Q - Q_max_local)
        exp_logits = prior_policy * np.exp(logits)
        pi = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)
        return pi

    policy = pi_from_Q(Q, beta, prior_policy)

    def get_mdp_generator(S, A, transition_dynamics_sparse, policy):
        rows, cols, data = [], [], []
        td = transition_dynamics_sparse.tocoo()
        for s_j, col, prob in zip(td.row, td.col, td.data):
            for a_j in range(A):
                row = s_j * A + a_j
                rows.append(row)
                cols.append(col)
                data.append(prob * policy[s_j, a_j])
        return csr_matrix((data, (rows, cols)), shape=(S * A, S * A))

    mdp_generator = get_mdp_generator(S, A, transition_dynamics_sparse, policy)
    Qj = (np.log(mdp_generator.dot(exp_beta_Q) + 1e-12) + beta * Q_max) / beta
    Qj = Qj.reshape(S, A)

    delta_rwd = rewards_table + gamma * Qj - Q
    delta_min = np.min(delta_rwd)
    delta_max = np.max(delta_rwd)

    lb = Q + delta_rwd + gamma * delta_min / (1 - gamma)
    ub = Q + delta_rwd + gamma * delta_max / (1 - gamma)

    r_min = np.min(rewards_table)
    r_max = np.max(rewards_table)
    lb = np.maximum(lb, r_min / (1 - gamma))
    ub = np.minimum(ub, r_max / (1 - gamma))

    return lb, ub

from scipy.sparse import csr_matrix

def q_learning(N_steps, test_steps):
    global Q
    epsilon = epsilon_initial
    episode_rewards = []
    state = start_state
    step = 1

    rewards_table = np.zeros((S, A))
    dynamics_table = np.zeros((S, A, S))
    L = np.full((S, A), -np.inf)
    U = np.full((S, A), np.inf)
    bound_update_freq = 5000

    while step < N_steps + 1:
        if state in terminal_state:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            state = start_state

        action = epsilon_greedy_policy(state, epsilon)
        next_state = np.random.choice(S, p=T[state, action])
        reward = r[state, action, next_state]

        # Update empirical model
        rewards_table[state, action] = reward
        dynamics_table[state, action, next_state] += 1

        # Update bounds every bound_update_freq steps
        if step % bound_update_freq == 0 and step > 5000:
            L, U = compute_bounds(Q, dynamics_table, rewards_table, gamma)

        # Q-learning update with bounds
        td_target = reward + discount_factor * np.max(Q[next_state])
        new_q = Q[state, action] + learning_rate * (td_target - Q[state, action])
        if L[state, action] > -np.inf and U[state, action] < np.inf:
            Q[state, action] = np.clip(new_q, L[state, action], U[state, action])
        else:
            Q[state, action] = new_q

        state = next_state
        step += 1

        if step % test_steps == 0:
            tr = test_q()
            episode_rewards.append(tr)

    return episode_rewards



# Define Q-learning parameters
learning_rate = 0.1
discount_factor = gamma
epsilon_initial = 1.0
epsilon_decay = 0.997
# if(nst>3):
#     epsilon_decay=0.996
epsilon_min = 0.01
# num_episodes = 4000
max_steps=S #13

N_steps=15000
test_steps = 2 #12
# N_steps=30000
# test_steps = 6 #12
num_episodes = int(N_steps/test_steps)

# Run multiple episodes and average results
num_runs = 30
average_rewards = np.zeros(num_episodes)
rewards_run = np.zeros((num_runs, num_episodes))
for run in range(num_runs):
    Q = np.zeros((S,A))
    # np.random.seed(run)
    episode_rewards = q_learning(N_steps, test_steps)
    rewards_run[run] = episode_rewards
    average_rewards += np.array(episode_rewards)
average_rewards /= num_runs
end_time = time.time()
np.save("sqb.npy", rewards_run)
w=100
# Plot average Q value per episode over 5 runs
plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.title(f'Average Reward per Episode over {num_runs} Runs')
plt.show()
