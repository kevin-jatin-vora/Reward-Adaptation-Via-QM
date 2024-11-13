import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
#np.random.seed(1)


class FrozenLake:
    def __init__(self, size=4):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.start = (0, 0)
        self.goal = (size - 1, 1)
        self.hole_prob = 0.1
        self.holes = [] 
        self.generate_holes()

    def generate_holes(self, num_pairs=4):
        pairs = set()
        while len(pairs) < num_pairs:
          pair = (np.random.randint(0, self.size), np.random.randint(0, self.size))
          if pair not in pairs and pair != self.goal and pair != self.start:
            pairs.add(pair)
        self.holes = list(pairs)
        for x,y in self.holes:
            self.grid[x, y] = -1

    def is_valid_position(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_reward1(self, state):
        x, y = state
        if (x, y) == self.holes[0] or (x, y) == self.holes[2]:
            return 1  
        elif(x, y) == self.holes[1] or (x, y) == self.holes[3]:
            return -1
        elif (x, y) == self.goal:
            return 0.6
        else:
            return -0.01  # For all other states

    def get_reward2(self, state):
        x, y = state
        if (x, y) == self.holes[1] or (x, y) == self.holes[3]:
            return 1  
        elif(x, y) == self.holes[0] or (x, y) == self.holes[2]:
            return -1
        elif (x, y) == self.goal:
            return 0.6 
        else:
            return -0.01  # For all other states

    def get_reward(self, state):
        x, y = state
        if (x, y) == self.goal:
            return 0.6*2  # Goal reached
        elif self.grid[x, y] == -1:
            return 0  # Fell into a hole
        else:
            return -0.02  # For all other states

    def get_neighboring_states(self, state):
        x, y = state
        neighbors = []
        for action in ['up', 'down', 'left', 'right']:
            new_x, new_y = self.transition(state, action)
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        return neighbors

    def transition(self, state, action):
        x, y = state
        if action == 'up':
            x -= 1
        elif action == 'down':
            x += 1
        elif action == 'left':
            y -= 1
        elif action == 'right':
            y += 1
        if self.is_valid_position(x, y):
            return x, y
        return state
              
def display_state(env, agent_location):
    state_grid = np.empty((env.size, env.size), dtype=object)
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == agent_location:
                state_grid[i, j] = 'A'  # Agent's current position
            elif (i, j) == env.start:
                state_grid[i, j] = 'S'  # Agent's start position
            elif (i, j) == env.goal:
                state_grid[i, j] = 'G'  # Goal position
            elif env.grid[i, j] == -1:
                state_grid[i, j] = 'H'  # Hole
            else:
                state_grid[i, j] = '-'  # Empty tile
    
    return state_grid

def make_transitions_stochastic(T1, x_percentage):
    S, A, _ = T1.shape
    
    # Calculate the number of states to remain deterministic
    num_deterministic_states = int(S * x_percentage/100)
    
    # Randomly select which states will remain deterministic
    deterministic_states = np.random.choice(S, num_deterministic_states, replace=False)
    
    # Initialize modified transition probability matrix
    T_stochastic = np.zeros_like(T1)
    
    # Modify transition probabilities for deterministic states
    for s in deterministic_states:
        for a in range(A):
            max_prob_index = np.argmax(T1[s, a])
            T_stochastic[s, a, max_prob_index] = 1
     
    # Modify transition probabilities for stochastic states
    stochastic_states = np.setdiff1d(np.arange(S), deterministic_states)
    for s in stochastic_states:
        for a in range(A):
            T_stochastic[s, a] = T1[s, a]
    return T_stochastic


def restrict_transition(matrix, max_bf):
    S, A, S_prime = matrix.shape

    # Sample n uniformly from [1, max_bf] for each state
    n_values = np.random.randint(1, max_bf + 1, size=S)
    print(n_values.mean())
    # Identify top n states for each action
    # top_n_indices = np.argsort(matrix, axis=2)[:, :, -n_values[:, None]]
    sorted_matrix = np.argsort(matrix, axis=2)

    # Create a mask to zero out probabilities for states not in top n
    mask = np.zeros_like(matrix)
    for s in range(S):
        for a in range(A):
            mask[s, a, sorted_matrix[s, a, -n_values[s]:]] = 1

    # Apply mask
    restricted_matrix = matrix * mask

    # Normalize probabilities for the top n states for each action
    row_sums = restricted_matrix.sum(axis=2, keepdims=True)
    normalized_matrix = restricted_matrix / row_sums

    return normalized_matrix
    
env = FrozenLake(size=6)  # Change size as needed
total_states = env.size * env.size
T = np.zeros((total_states, 4, total_states))

def state_to_xy(state, size=env.size):
    x = state // size
    y = state % size
    return x, y

for s in range(total_states):
    for a in range(4):
        ns = env.get_neighboring_states(state_to_xy(s))
        sdash_true = [np.ravel_multi_index(env.transition((s // env.size, s % env.size), ['up', 'down', 'left', 'right'][a]), (env.size, env.size))][0]
        for sdash in ns:
            sd = sdash[0] * env.size + sdash[1]
            if(sd == sdash_true):
                T[s,a,sd] += 0.79 #0.7
            else:
                T[s,a,sd] += 0.07 #0.1

holes = env.holes.copy()
terminal_state = holes + [env.goal]

def save_env(w_filename, t_filename, env, T):
    np.save(w_filename,env.grid)
    np.save(t_filename, T)

data = []
init=int(input("enter start index: "))
till=int(input("enter end index: "))
for avg in range(init,till): #for avg in range(30):
    if(not os.path.isdir(str(avg))):
        os.mkdir(str(avg))
    # np.random.seed(avg)
    env = FrozenLake(size=6) 
    holes = env.holes.copy()
    terminal_state = holes + [env.goal]
    # f = open(f"{avg}//env_QL.txt", "w")
    # f.write(f"{display_state(env, 0)}")
    # f.close()
    for nst in range(1, 5):
        # if(nst<=2):
        #     continue
        # np.random.seed(avg)
        ts = [state[0] * env.size + state[1] for state in terminal_state]
        x_percentage = 0
        T1=T.copy()
        T1 = make_transitions_stochastic(T1, x_percentage)
        number_stochastic_transitions = nst
        T1 = restrict_transition(T1, number_stochastic_transitions)
        T1[ts, :, :] = 0
        gamma = 0.89
        save_env(f"{avg}//env_{nst}.npy", f"{avg}//T_{nst}.npy", env, T1)
        start_time = time.time()
    
        def epsilon_greedy_policy(state, epsilon, Q):
            if np.random.rand() < epsilon:
                return np.random.choice(range(len(Q[state])))
            else:
                return np.argmax(Q[state])
        
        def test_q(e=30):
            global Q
            episode_rewards=[]
            for episode in range(e):
                state = 0
                total_reward = 0
                step = 0
                while state not in ts and step<max_steps:
                    step+=1
                    action = np.argmax(Q[state])
                    next_state = np.random.choice(total_states, p=T1[state, action, :])
                    reward = env.get_reward((next_state // env.size, next_state % env.size))
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
            return np.mean(episode_rewards) 
        
        def q_learning(N_steps, test_steps, env, learning_rate, discount_factor, epsilon_initial, epsilon_decay, epsilon_min, num_episodes):
            num_actions = 4
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = 0
            step = 1
            
            while step < N_steps+1:
                if state in ts:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    # Reset to the initial state if the agent reaches a terminal state
                    state = 0
        
                action = epsilon_greedy_policy(state, epsilon,Q)
                next_state = np.random.choice(total_states, p=T1[state, action, :])
                reward = env.get_reward((next_state // env.size, next_state % env.size))
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
        discount_factor = gamma #0.9
        epsilon_initial = 1.0
        epsilon_decay = 0.9999#8
        epsilon_min = 0.01
        # num_episodes = 2000
        max_steps = 20
        
        N_steps=4800*4
        test_steps = 10 #12
        num_episodes = int(N_steps/test_steps)
        
        # Run multiple episodes and average results
        num_runs = 30
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
    
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):
            Q=np.zeros((total_states,4))
            #np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps,env, learning_rate, discount_factor, epsilon_initial, epsilon_decay, epsilon_min, num_episodes)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
    
        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//QL_{x_percentage}_{number_stochastic_transitions}.csv')
        # Plot average Q value per episode over 5 runs
        window_size = 100
        # plt.plot(range(num_episodes - window_size + 1), np.convolve(average_rewards, np.ones(window_size), 'valid') / window_size)
        # plt.xlabel('Episode')
        # plt.ylabel('Average Reward')
        # plt.title(f'Average Reward per Episode over {num_runs} Runs')
        # plt.show()
        # data.append((avg, f'{x_percentage}_{number_stochastic_transitions}', end_time - start_time))
pd.DataFrame(data, columns=["Run", "Domain info", "QL"]).to_csv(f"Data_QL_{till-1}.csv")
