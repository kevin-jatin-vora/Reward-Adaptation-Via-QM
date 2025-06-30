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


def q_learning(N_steps, test_steps):
    global Q
    epsilon = epsilon_initial
    episode_rewards = []
    state = start_state
    step = 1

    while step < N_steps+1:
        if state in terminal_state:
            # Decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            # Reset to the initial state if the agent reaches a terminal state
            state = start_state

        action = epsilon_greedy_policy(state, epsilon)
        next_state = np.random.choice(S, p=T[state, action, :])
        reward = r[state, action, next_state]
        # Update Q-value
        Q[state, action] += learning_rate * (
            reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
        )

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
np.save("QL_traditonal.npy", rewards_run)
w=100
# Plot average Q value per episode over 5 runs
plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.title(f'Average Reward per Episode over {num_runs} Runs')
plt.show()
