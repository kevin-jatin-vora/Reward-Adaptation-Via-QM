import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
#np.random.seed(1)
import argparse
import yaml

# Load config
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Common parameters
init = cfg['start']
till = cfg['end']
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

env = FrozenLake(size=6)  # Change size as needed
total_states = env.size * env.size
T = np.zeros((total_states, 4, total_states))

def state_to_xy(state, size=env.size):
    x = state // size
    y = state % size
    return x, y


# holes = env.holes.copy()
# terminal_state = holes + [env.goal]
def get_env(w_filename, t_filename):
    gridworld = np.load(w_filename)
    env = FrozenLake(size=6)
    env.grid = gridworld
    env.size = gridworld.shape[0]
    env.start = (0, 0)
    env.goal = (env.size - 1, 1)
    a=np.where(env.grid==-1)
    env.holes = list(zip(a[0],a[1]))
    holes = env.holes
    terminal_state = holes + [env.goal]
    T = np.load(t_filename)
    return env,T, terminal_state
A=4
data = []
for avg in range(init,till):
    print(avg)
    for nst in nst_range:
        env, T1, terminal_state = get_env(f"{10}//env_{nst}.npy", f"{10}//T_{nst}.npy")
        ts = [state[0] * env.size + state[1] for state in terminal_state]
        number_stochastic_transitions = nst
        start_time = time.time()
        T1[ts, :, :] = 0
        
        def compute_q_values(S, A, R, gamma, T=T1):
            # Initialize Q-values to zeros
            Q_new = np.zeros((S, A))
            
            # Maximum number of iterations for value iteration
            max_iterations = 5000
            
            # Value iteration
            for _ in range(max_iterations):
                Q = Q_new.copy()
                for s in range(S):
                    for a in range(A):
                        q_sa = 0
                        for s_prime in range(S):
                            q_sa += T[s][a][s_prime] * (eval(R.replace('()',f'({(s_prime // env.size, s_prime % env.size)})')) + gamma * np.max(Q[s_prime]))
                        Q_new[s][a] = q_sa
                if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
                    print("Converged in", _ + 1, "iterations")
                    break
                Q = Q_new
            
            return Q
        
        # Assuming read_mdp function is defined as mentioned in the question
        #S, A, R, R2, T, gamma, terminal_state = read_mdp("mdp_exp1.txt")
        
        # Compute Q-values
        q_p1 = compute_q_values(env.size*env.size, 4, 'env.get_reward1()', gamma)
        
        # Compute Q-values
        q_p2 = compute_q_values(env.size*env.size, 4, 'env.get_reward2()', gamma)
        
        # np.save(f'Q1_{nst}.npy',q_p1)
        # np.save(f'Q2_{nst}.npy',q_p2)
        # np.save(f'Q_{nst}.npy',compute_q_values(env.size,env.size, 4, 'env.get_reward()', gamma))
        
        def compute_q_values_mu(S, A, R, gamma, T=T1):
            # Initialize Q-values to zeros
            Q_new = np.zeros((S, A))
            
            # Maximum number of iterations for value iteration
            max_iterations = 5000
            
            # Value iteration
            for _ in range(max_iterations):
                Q = Q_new.copy()
                for s in range(S):
                    for a in range(A):
                        q_sa = 0
                        for s_prime in range(S):
                            q_sa += T[s][a][s_prime] * (eval(R.replace('()',f'({(s_prime // env.size, s_prime % env.size)})')) + gamma * np.min(Q[s_prime]))
                        Q_new[s][a] = q_sa
                if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
                    print("Converged in", _ + 1, "iterations")
                    break
                Q = Q_new
            
            return Q
        # Compute Q-values
        q_m1 = compute_q_values_mu(env.size*env.size, 4, 'env.get_reward1()', gamma)
        # Compute Q-values
        q_m2 = compute_q_values_mu(env.size*env.size, 4, 'env.get_reward2()', gamma)

        
        #np.save(f"{avg}//Q1_{nst}.npy", q_p1)
        #np.save(f"{avg}//Q2_{nst}.npy", q_p2)
        
        def transition(current_state, action):
            global T1, number_stochastic_transitions
            p=T1[current_state, action]
            # if(np.sum(p)==1):
            t=np.argsort(p)[-number_stochastic_transitions:]
            return t[np.where(T1[current_state,action,np.argsort(T1[current_state,action])[-number_stochastic_transitions:]]!=0)]

        
        start_time = time.time()
        Q = q_p1 + q_p2 
        # Q_UB = Q.copy()
        Q[ts,:] = 0
        for i in range(5000):
            if(i>0):
                U = Q_k.copy()
                Udash = Q.copy()
            Q_k=Q.copy()
            for s in range(total_states):
                if(s in ts):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition(s,a):
                            temp.append(env.get_reward(state_to_xy(sdash)) + gamma*np.max(Q_k[sdash]))
                        Q[s,a] =min(Q_k[s,a],max(temp)) #max(temp)             
            # if(i>0):
            #     if(np.round(np.max(np.abs(Q_k-Q)),7)> np.round(gamma*(np.max(np.abs(U-Udash))),7)):
            #         # print(np.max(np.abs(Q_k-Q)))
            #         # print(gamma*(np.max(np.abs(U-Udash))))
            #         print(i)
            #         # input()
            if(np.max(np.abs(Q-Q_k))<1e-12): 
                print(i)
                print("------------")
                break
        
        
        Qm = np.zeros((total_states,A))
        o1 = q_p1 + q_m2
        o2 = q_p2 + q_m1
        
        for s in range(total_states):
            for a in range(A):
                Qm[s,a]= max(o1[s,a], o2[s,a])
        
        
        Qm[ts,:] = 0
        for i in range(5000):
            if(i>0):
                U = Qm_k.copy()
                Udash = Qm.copy()
            Qm_k=Qm.copy()
            for s in range(total_states):
                if(s in ts):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition(s,a):
                            temp.append(env.get_reward(state_to_xy(sdash)) + gamma*np.max(Qm_k[sdash]))
                        Qm[s,a] = max(Qm_k[s,a],min(temp)) #min(temp)
            # if(i>0):
            #     if(np.max(np.abs(Qm_k-Qm))> gamma*(np.max(np.abs(U-Udash)))):
            #         # print("lowerbound")
            #         print(i)
            #         # input()
            if(np.max(np.abs(Qm-Qm_k))<1e-12):
                print(i)
                print("------------")
                break
        
        
        info=[]
        final_actions=set(list(range(A)))
        prune={}
        state_action={}
        
        
        for i in range(total_states):
            alist=[]
            for action_l in range(A):
                for action_u in range(A):
                    if(action_l==action_u):
                        continue
                    if( Qm[i, action_l]-Q[i,action_u] > 1e-7 ):
                        info.append((i,action_l, action_u))
                        alist.append(action_u)
            prune[i]= set(alist)
            state_action[i]= final_actions.difference(set(alist))
            
        # print(total_states*A-sum([len(state_action[i]) for i in state_action.keys()]))
        
        ########################################################################################
        violations1=[]
        violations2=[]
        def epsilon_greedy_policy(state, epsilon, Q):
            global violations1, violations2
            if np.random.rand() < epsilon:
                return np.random.choice(list(state_action[state]))
            else:
                best_actions = np.where(Q[state] == np.max(Q[state]))[0]
                if(len(best_actions)>1):
                    try:
                        action = np.random.choice(list(set(state_action[state]).intersection(set(list(best_actions)))))
                    except:
                        # action = np.random.choice(list(state_action[state]))
                        state_index = state
                        q_values = Q[state_index, list(state_action[state_index])]
                        best_action_indices = np.where(q_values == np.max(q_values))[0]
                        best_action_index = np.random.choice(best_action_indices)
                        bas = list(state_action[state_index])
                        action = bas[best_action_index]
                else:
                    action=best_actions[0]
                if(action  not in list(state_action[state])):
                    if(epsilon==0):
                        violations1.append((avg, state, display_state(env, state), state_action[state], Q[state].copy(),(action, epsilon)))
                    else:
                        violations2.append((avg, state, display_state(env, state), state_action[state], Q[state].copy(),(action, epsilon)))
                    state_index = state
                    q_values = Q[state_index, list(state_action[state_index])]
                    best_action_indices = np.where(q_values == np.max(q_values))[0]
                    best_action_index = np.random.choice(best_action_indices)
                    bas = list(state_action[state_index])
                    action = bas[best_action_index]
                return action
       
        
        def test_q(e=30):
            global Q
            episode_rewards=[]
            for episode in range(e):
                state = 0
                total_reward = 0
                step = 0
                while state not in ts and step<max_steps:
                    step+=1
                    action = epsilon_greedy_policy(state, 0, Q)
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
        
        discount_factor = gamma
        
        num_episodes = int(N_steps/test_steps)
        
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        
        for run in range(num_runs):
            Q = np.zeros((total_states,A))
            #np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps,env, learning_rate, discount_factor, epsilon_initial, epsilon_decay, epsilon_min, num_episodes)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        
        average_rewards /= num_runs
        end_time = time.time()
        