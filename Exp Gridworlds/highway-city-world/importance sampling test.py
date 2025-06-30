import numpy as np
from matplotlib import pyplot as plt
# import pandas as pd
# import time

S=49
A = 4
terminal_state = np.load("terminal.npy")
start_state = np.load("initial.npy")[0]

R1, R2 = np.load("R1.npy"),np.load("R2.npy") #,np.load("center_coords.npy")#create_stochastic_taxi_matrices(n=n, success_prob=0.8)
# r=R1+R2
r=R2
T=np.load(r"T.npy")

def epsilon_greedy(Q_star, Q_mu, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, A)  # Random action
    else:
        # Select one action from Q_star using argmax and another from Q_mu using argmin
        action_star = np.argmax(Q_star[state])  # Action that maximizes Q_star
        action_mu = np.argmin(Q_mu[state])  # Action that minimizes Q_mu
        
        # Randomly return one of these two actions
        return np.random.choice([action_star, action_mu], p=[0.5,0.5])


# # Helper function: epsilon-greedy action selection
def epsilon_greedy_og(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, A)  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action

# Function to test policy
def test_policy(Q, episodes, max_steps):
    total_return = 0
    for _ in range(episodes):
        state = start_state  # Start from the initial state
        episode_return = 0
        steps = 0
        while state not in terminal_state and steps < max_steps:
            steps += 1
            action = np.argmax(Q[state])  # Greedy action
            next_state = np.random.choice(range(S), p=T[state, action])
            reward = r[state, action, next_state]
            episode_return += reward
            state = next_state
        total_return += episode_return
    return total_return / episodes

def test_policy_mu(Q, episodes, max_steps):
    total_return = 0
    for _ in range(episodes):
        state = start_state  # Start from the initial state
        episode_return = 0
        steps = 0
        while state not in terminal_state and steps < max_steps:
            steps += 1
            action = np.argmin(Q[state])  # Greedy action
            next_state = np.random.choice(range(S), p=T[state, action])
            reward = r[state, action, next_state]
            episode_return += reward
            state = next_state
        total_return += episode_return
    return total_return / episodes


alpha = 0.1
gamma = 0.9
epsilon_initial = 1.0
epsilon_decay = 0.997 #0.992
epsilon_min = 0.01
max_steps = S
num_runs = 1
N_steps=15000
test_steps = 2 #12
num_episodes = int(N_steps/test_steps)

avg_returns_Q_og_all_runs = []
avg_returns_Q_star_all_runs = []
avg_returns_Q_mu_all_runs = []

for run in range(num_runs):
    # print(f"Run {run + 1}/{num_runs}")
    # Reset Q-tables and other parameters for each run
    Q_og = np.zeros((S, A))
    Q_star = np.zeros((S, A))
    Q_mu = np.zeros((S, A))

    test_avg_returns_Q_og = []
    test_avg_returns_Q_star = []
    test_avg_returns_Q_mu = []

    # Training and testing for Q_star and Q_mu
    epsilon = epsilon_initial
    state = start_state
    steps = 0
    done = False
    ep_step = 0
    while steps < N_steps + 1:
        ep_step += 1
        # if state in terminal_state:
        if done:
            done = False
            ep_step = 0
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            while state in terminal_state:
                state = start_state
        steps += 1
        action = epsilon_greedy(Q_star, Q_mu, state, epsilon)
        next_state = np.random.choice(range(S), p=T[state, action])
        reward = r[state, action, next_state]
        
        # Compute importance sampling ratios
        a_star = np.argmax(Q_star[state])
        a_mu = np.argmin(Q_mu[state])

        # Probability under behavior policy
        b_action = (epsilon / A) + ((1 - epsilon) * 0.5 if action in [a_star, a_mu] else 0)

        # Probability under target policy for Q_star
        pi_star_action = (epsilon / A) + (1 - epsilon if action == a_star else 0)
        w_star = pi_star_action / b_action

        # Probability under target policy for Q_mu
        pi_mu_action = (epsilon / A) + (1 - epsilon if action == a_mu else 0)
        w_mu = pi_mu_action / b_action

        # Update Q_star and Q_mu using weighted updates
        best_next_action = np.argmax(Q_star[next_state])
        Q_star[state, action] += w_star * alpha * (reward + gamma * Q_star[next_state, best_next_action] - Q_star[state, action])

        worst_next_action_value = np.min(Q_mu[next_state])
        Q_mu[state, action] += w_mu * alpha * (reward + gamma * worst_next_action_value - Q_mu[state, action])

        state = next_state
        if state in terminal_state or ep_step>max_steps:
            done=True
    

        if steps % test_steps == 0:
            avg_return_Q_star = test_policy(Q_star, 1, max_steps)
            test_avg_returns_Q_star.append(avg_return_Q_star)
            avg_return_Q_mu = test_policy_mu(Q_mu, 1, max_steps)
            test_avg_returns_Q_mu.append(avg_return_Q_mu)

    # Training and testing for Q_og
    epsilon = epsilon_initial
    state = start_state
    steps = 0
    done = False
    ep_step = 0
    while steps < N_steps + 1:
        ep_step += 1
        # if state in terminal_state:
        if done:
            done = False
            ep_step = 0
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            while state in terminal_state:
                state = start_state
        steps += 1
        action = epsilon_greedy_og(Q_og, state, epsilon)
        next_state = np.random.choice(range(S), p=T[state, action])
        reward = r[state, action, next_state]
    
        best_next_action = np.argmax(Q_og[next_state])
        Q_og[state, action] += alpha * (reward + gamma * Q_og[next_state, best_next_action] - Q_og[state, action])
        state = next_state
        if state in terminal_state or ep_step>max_steps:
            done=True
    
        if steps % test_steps == 0:
            avg_return_Q_og = test_policy(Q_og, 1, max_steps)
            test_avg_returns_Q_og.append(avg_return_Q_og)

    # Append results for this run
    avg_returns_Q_og_all_runs.append(test_avg_returns_Q_og)
    avg_returns_Q_star_all_runs.append(test_avg_returns_Q_star)
    avg_returns_Q_mu_all_runs.append(test_avg_returns_Q_mu)

# Calculate the average returns across all runs
avg_returns_Q_og = np.mean(avg_returns_Q_og_all_runs, axis=0)
avg_returns_Q_star = np.mean(avg_returns_Q_star_all_runs, axis=0)
avg_returns_Q_mu = np.mean(avg_returns_Q_mu_all_runs, axis=0)

# Plot the results
# x_ticks = range(test_steps, len(avg_returns_Q_star) * test_steps + 1, test_steps)
l=-1000
plt.figure(figsize=(8.5, 6))
window_size = 100
# plt.plot(avg_returns_Q_og[:l], label="Q* (epsilon-greedy)", alpha=0.7)
plt.plot(range(num_episodes - window_size+l+1), np.convolve(avg_returns_Q_og[:l], np.ones(window_size), 'valid') / window_size, label="Q* (epsilon-greedy)")
plt.plot(range(num_episodes - window_size+l+1), np.convolve(avg_returns_Q_star[:l], np.ones(window_size), 'valid') / window_size, label="Q* (modified epsilon-greedy)")
plt.plot(range(num_episodes - window_size+l+1), np.convolve(avg_returns_Q_mu[:l], np.ones(window_size), 'valid') / window_size, label="Q_mu (modified epsilon-greedy)")

# Create the legend with 3 columns
# handles, labels = plt.gca().get_legend_handles_labels()  # Get handles and labels
# plt.legend(handles, labels, ncol=1, loc='upper center', bbox_to_anchor=(1.5, 0.5)) # Adjust bbox_to_anchor for fine-tuning position
plt.legend(loc='center right')
plt.xlabel("Steps")
plt.ylabel("Average Return")
# plt.legend()
plt.grid()
# plt.title(f"Average Returns Over {num_runs} Runs")

# fig.text(0.5, 0.014, 'Step', ha='center', va='center')
# fig.text(0.014, 0.5, 'Average Return', ha='center', va='center', rotation='vertical')
plt.rcParams['font.size'] = '20'

plt.tight_layout()
plt.savefig('R2_Qstar-Qmu.png', bbox_inches='tight', dpi=600)


plt.show()
