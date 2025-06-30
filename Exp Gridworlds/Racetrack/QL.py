import numpy as np
#np.random.seed(6)
from matplotlib import pyplot as plt
import time
import os
import argparse
import yaml

# --- Config loading ---
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# --- Common Config Variables ---
init = cfg['start']
last = cfg['end']
nst_range = cfg['nst_range']
gamma = cfg['gamma']
x_percentage = cfg['x_percentage']

learning_rate = cfg['learning_rate']
epsilon_initial = cfg['epsilon_initial']
epsilon_decay = cfg['epsilon_decay']
epsilon_min = cfg['epsilon_min']
max_steps = cfg['max_steps']
N_steps = cfg['N_steps']
test_steps = cfg['test_steps']

WORLD = np.array([
    ["G", "_", "_", "_", "_", "X", "X"],
    ["G", "_", "_", "_", "_", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["X", "X", "X", "X", "X", "_", "_"],
    ["X", "X", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_"],
    ["S", "_", "_", "_", "X", "X", "X"]
])
S=WORLD.size
A=7
STATES = range(WORLD.size)  # 1D array from 0 to 48
STATE2WORLD = {
    0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6),
    7: (1, 0), 8: (1, 1), 9: (1, 2), 10: (1, 3), 11: (1, 4), 12: (1, 5), 13: (1, 6),
    14: (2, 0), 15: (2, 1), 16: (2, 2), 17: (2, 3), 18: (2, 4), 19: (2, 5), 20: (2, 6),
    21: (3, 0), 22: (3, 1), 23: (3, 2), 24: (3, 3), 25: (3, 4), 26: (3, 5), 27: (3, 6),
    28: (4, 0), 29: (4, 1), 30: (4, 2), 31: (4, 3), 32: (4, 4), 33: (4, 5), 34: (4, 6),
    35: (5, 0), 36: (5, 1), 37: (5, 2), 38: (5, 3), 39: (5, 4), 40: (5, 5), 41: (5, 6),
    42: (6, 0), 43: (6, 1), 44: (6, 2), 45: (6, 3), 46: (6, 4), 47: (6, 5), 48: (6, 6)
}

START = 42  # state index of start
GOALS = [0, 7]  # state index of goals
WALLS = [  # state index of walls
    5, 6,
    14, 15,
    21, 22, 23, 24, 25,
    28, 29,
    46, 47, 48
]

CRASH = -10.  # reward for hitting a wall
WIN = 100.  # reward for reaching a goal
STEP = -1.  # reward for moving

ACTIONS = [  # set of all actions
    (0, 0), 
    (-1, 0), (-2, 0), #up
    (0, 1), (0, 2), #right
    (0, -1), (0, -2), #left
]
R=np.zeros((S,A,S))
R2=np.zeros((S,A,S))
R3=np.zeros((S,A,S))
for s in range(S):
    for a in range(A):
        for sdash in range(S):
            if(sdash in GOALS):
                R3[s,a,sdash] = 2
            if(sdash not in WALLS and sdash not in GOALS):
                R2[s,a,sdash] = 0.2#0.9
                R3[s,a,sdash] = -0.3#-1
            if (sdash in WALLS):
                R2[s,a,sdash] = -0.5#-2.5
                R3[s,a,sdash] = 0.3#2.3
            if(sdash==42):
                R[s,a,sdash] = 3
                R3[s,a,sdash] = -4

r=R+R2+R3

def transition(state,action):
    dx, dy = ACTIONS[action]
    next_state = (STATE2WORLD[state][0] +dx, STATE2WORLD[state][1] +dy)
    if(next_state[0]*7 + next_state[1] in WALLS or next_state[0]<0 or next_state[0]>7 or next_state[1]<0 or next_state[1]>7):
        return next_state[0]*7 + next_state[1], False
    else:
        return next_state[0]*7 + next_state[1], True

def neighbours(s):
    n=[]
    for a in range(7):
        ns = transition(s,a)
        if(ns[1]):
            n.append(ns[0])
        else:
            if(ns[0] in STATE2WORLD):
                n.append(ns[0])
            else:
                n.append(s)
    return list(set(n))

T = np.zeros((49,7,49))

for s in range(49):
    if(s in WALLS or s in GOALS):
        continue
    reachable = neighbours(s)
    ns=[]
    for i in reachable:
        if(i in STATE2WORLD):
            ns.append(i)
        else:
            ns.append(s)
    for a in range(7):
        nxt_s = list(transition(s,a))
        if(nxt_s[0] not in STATE2WORLD):
            nxt_s[0] = s
        for sdash in range(49):
            if(nxt_s[0]==sdash):
                T[s,a,sdash] = 0.88
            elif(sdash in ns):
                T[s,a,sdash] = 0.12/(len(ns)-1)
            else:
                T[s,a,sdash] = 0
        if(np.sum(T[s,a])!=1):
            T[s,a,nxt_s[0]] += 1 - np.sum(T[s,a])


# def make_transitions_stochastic(T1, x_percentage):
#     S, A, _ = T1.shape
    
#     # Calculate the number of states to remain deterministic
#     num_deterministic_states = int(S * x_percentage/100)
    
#     # Randomly select which states will remain deterministic
#     deterministic_states = np.random.choice(S, num_deterministic_states, replace=False)
    
#     # Initialize modified transition probability matrix
#     T_stochastic = np.zeros_like(T1)
    
#     # Modify transition probabilities for deterministic states
#     for s in deterministic_states:
#         for a in range(A):
#             max_prob_index = np.argmax(T1[s, a])
#             T_stochastic[s, a, max_prob_index] = 1
     
#     # Modify transition probabilities for stochastic states
#     stochastic_states = np.setdiff1d(np.arange(S), deterministic_states)
#     for s in stochastic_states:
#         for a in range(A):
#             T_stochastic[s,a] = T1[s,a]
#     return T_stochastic

# def restrict_transition(matrix, max_bf):
#     S, A, S_prime = matrix.shape

#     # Sample n uniformly from [1, max_bf] for each state
#     n_values = np.random.randint(1, max_bf + 1, size=S)
#     print(n_values.mean())
#     # Identify top n states for each action
#     # top_n_indices = np.argsort(matrix, axis=2)[:, :, -n_values[:, None]]
#     sorted_matrix = np.argsort(matrix, axis=2)

#     # Create a mask to zero out probabilities for states not in top n
#     mask = np.zeros_like(matrix)
#     for s in range(S):
#         for a in range(A):
#             mask[s, a, sorted_matrix[s, a, -n_values[s]:]] = 1

#     # Apply mask
#     restricted_matrix = matrix * mask

#     # Normalize probabilities for the top n states for each action
#     row_sums = restricted_matrix.sum(axis=2, keepdims=True)
#     normalized_matrix = restricted_matrix / row_sums

#     return normalized_matrix

data = []
for avg in range(init,last):
    for nst in nst_range:
        x_percentage = 0
        # T1 = make_transitions_stochastic(T, x_percentage)
        number_stochastic_transitions = nst #A
        # T1 = restrict_transition(T1, number_stochastic_transitions)
        # T1[GOALS,:, :] = 0
        # T1[WALLS,:, :] = 0
        gamma=gamma
        filename = f'{avg}//T_{nst}.npy'
        T1=np.load(filename)
        
        start_time = time.time()
        # Initialize Q-table
        Q = np.zeros((S, A))
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.choice(A)
            else:
                return np.argmax(Q[state])
        
        def test_q(e=30):
            global Q
            episode_rewards=[]
            for episode in range(e):
                state = START
                total_reward = 0
                step = 0
                while state not in WALLS and state not in GOALS and step<max_steps:
                    step+=1
                    action = np.argmax(Q[state])
                    next_state = np.random.choice(S, p=T1[state, action, :])
                    reward = r[state, action, next_state]
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
            return np.mean(episode_rewards)
        
        def q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = START
            step = 1
        
            while step < N_steps+1:
                if state in WALLS or state in GOALS:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    # Reset to the initial state if the agent reaches a terminal state
                    state = START
        
                action = epsilon_greedy_policy(state, epsilon)
                next_state = np.random.choice(S, p=T1[state, action, :])
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
        discount_factor = gamma

        num_episodes = int(N_steps/test_steps)
        
        epsilon_min = 0.01
        
        
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):
            Q = np.ones((S,A))*-0.5
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        average_rewards /= num_runs
        end_time = time.time()
        