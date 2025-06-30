import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

init = cfg['start']
last = cfg['end']
gamma = cfg['gamma']
discount_factor = gamma
terminal_state = cfg['terminal_state']
x_percentage = cfg['x_percentage']
nst_range = cfg['nst_range']
N_steps = cfg.get('N_steps', 18000)
max_steps = cfg.get('max_steps', 30)
test_steps = cfg.get('test_steps', 12)
# Q-learning params
learning_rate = cfg.get('learning_rate', 0.1)
epsilon_initial = cfg.get('epsilon_initial', 1.0)
epsilon_decay = cfg.get('epsilon_decay', 0.999)
epsilon_min = cfg.get('epsilon_min', 0.01)

from scipy import stats
import random
import os
import numpy as np
#np.random.seed(6)
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.chdir(os.getcwd())

def read_mdp(mdp):

    """Function to read MDP file"""
    #mdp="mdp_new.txt"
    f = open(mdp)

    S = int(f.readline())
    A = int(f.readline())

    # Initialize Transition and Reward arrays
    R = np.zeros((S, A, S))
    R2 = np.zeros((S, A, S))
    T = np.zeros((S, A, S))

    # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R[s][a][sPrime] = line[sPrime]
                
     # Update the Reward Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            for sPrime in range(S):
                R2[s][a][sPrime] = line[sPrime]
    
    # Update the Transition Function
    for s in range(S):
        for a in range(A):
            line = f.readline().split()
            #print((s,a))
            for sPrime in range(S):
                #print(line[sPrime], end=" ")
                T[s][a][sPrime] = line[sPrime]
                 
            #print()

    # Read the value of gamma
    gamma = float(f.readline().rstrip())
    #terminal_state=int((f.readline().rstrip()))

    f.close()

    return S, A, R, R2, T, gamma#,terminal_state

S, A, R, R2, T, gamma = read_mdp("gridworld_mdp(r1+r2).txt")
R[31,1,40] = 0.6
R2[31,1,40] = 0.6
R[41,2,40] = 0.6
R2[41,2,40] = 0.6
R[39,3,40] = 0.6
R2[39,3,40] = 0.6

data=[]

data=[]
for avg in range(init, last):
    #print(avg)
    for nst in nst_range:
        number_stochastic_transitions = nst #A
        filename = f'{avg}//T_{nst}.npy'
        T1=np.load(filename)
  
        def compute_q_values(S, A, R, T, gamma, terminal_state):
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
                            q_sa += T[s][a][s_prime] * (R[s][a][s_prime] + gamma * np.max(Q[s_prime]))
                        Q_new[s][a] = q_sa
                if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
                    print("Converged in", _ + 1, "iterations")
                    break
                Q = Q_new
            
            return Q
        
        # Assuming read_mdp function is defined as mentioned in the question
        #S, A, R, R2, T, gamma, terminal_state = read_mdp("mdp_exp1.txt")
        
        # Compute Q-values
        q_p1 = compute_q_values(S, A, R, T1, gamma, terminal_state)
        
        # Compute Q-values
        q_p2 = compute_q_values(S, A, R2, T1, gamma, terminal_state)
        
        # Qstar = compute_q_values(S, A, R+R2, T1, gamma, terminal_state)
        
        def compute_q_values_mu(S, A, R, T, gamma, terminal_state):
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
                            q_sa += T[s][a][s_prime] * (R[s][a][s_prime] + gamma * np.min(Q[s_prime]))
                        Q_new[s][a] = q_sa
                if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
                    print("Converged in", _ + 1, "iterations")
                    break
                Q = Q_new
            
            return Q
        # Compute Q-values
        q_m1 = compute_q_values_mu(S, A, R, T1, gamma, terminal_state)
        # Compute Q-values
        q_m2 = compute_q_values_mu(S, A, R2, T1, gamma, terminal_state)
        
        
        def transition(current_state, action):
            global T1, S, number_stochastic_transitions
            p=T1[current_state, action]
            t=np.argsort(p)[-number_stochastic_transitions:]
            return t[np.where(T1[current_state,action,np.argsort(T1[current_state,action])[-number_stochastic_transitions:]]!=0)]

        
        start_time = time.time()
        
        real_r = R+R2
        
        Q = q_p1 + q_p2 
        # Q_UB = Q.copy()
        Q[terminal_state,:] = 0
        for i in range(5000):
            if(i>0):
                U = Q_k.copy()
                Udash = Q.copy()
            Q_k=Q.copy()
            for s in range(S):
                if(s in terminal_state):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition(s,a):
                            temp.append(real_r[s,a,sdash] + gamma*np.max(Q_k[sdash]))
                        Q[s,a] =min(Q_k[s,a],max(temp)) #max(temp)             

            if(np.max(np.abs(Q-Q_k))<0.0000000001): 
                print(i)
                print("------------")
                break
        
        Qm = np.zeros((S,A))
        o1 = q_p1 + q_m2
        o2 = q_p2 + q_m1
        
        for s in range(S):
            for a in range(A):
                Qm[s,a]= max(o1[s,a], o2[s,a])
        
        
        Qm[terminal_state,:] = 0
        for i in range(5000):
            if(i>0):
                U = Qm_k.copy()
                Udash = Qm.copy()
            Qm_k=Qm.copy()
            for s in range(S):
                if(s in terminal_state):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition(s,a):
                            temp.append(real_r[s,a,sdash] + gamma*np.max(Qm_k[sdash]))
                        Qm[s,a] = max(Qm_k[s,a],min(temp)) #min(temp)

            if(np.max(np.abs(Qm-Qm_k))<0.0000000001):
                print(i)
                print("------------")
                break
        
        
        info=[]
        final_actions=set(list(range(A)))
        prune={}
        state_action={}
        
        
        for i in range(S):
            alist=[]
            for action_l in range(A):
                for action_u in range(A):
                    if(action_l==action_u):
                        continue
                    if( Qm[i, action_l]-Q[i,action_u] > 1e-12 ):
                        info.append((i,action_l, action_u))
                        alist.append(action_u)
            prune[i]= set(alist)
            state_action[i]= final_actions.difference(set(alist))
            

        ########################################################################################
        rr=R+R2
    
        # Define environment parameters
        num_rows = 5
        num_cols = 9
        num_states = num_rows * num_cols
        num_actions = 4
        start_state = (0,4)
        goal_states = [(0, 0), (0, 8), (4, 4)]
        
        # Define rewards
        rewards = np.ones((num_rows, num_cols))*-0.0001
        rewards[0, 0] = 1.0
        rewards[0, 8] = 1.0
        rewards[4, 4] = 1.2
        
        # Define actions: down, up, left, right
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize Q-table
        Q = np.zeros((num_states, num_actions))
        
        # Convert state from (row, col) to index
        def state_to_index(state):
            row, col = state
            return row * num_cols + col
        
        # Convert index to state (row, col)
        def index_to_state(index):
            row = index // num_cols
            col = index % num_cols
            return row, col
        
        # Epsilon-greedy policy
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                # Exploration: Choose a random action from the available set
                return np.random.choice(list(state_action[state_to_index(state)]))
            else:
                # Exploitation: Choose a best action from the available set, breaking ties randomly
                state_index = state_to_index(state)
                q_values = Q[state_index, list(state_action[state_index])]
                best_action_indices = np.where(q_values == np.max(q_values))[0]
                best_action_index = np.random.choice(best_action_indices)
                bas = list(state_action[state_index])
                best_action = bas[best_action_index]
                return best_action
        
        def test_q(e=30):
            global Q
            episode_rewards = []
            for episode in range(e):
                state = start_state
                total_reward = 0
                step = 0
                while state not in goal_states and step<max_steps:
                    step+=1
                    action = epsilon_greedy_policy(state, 0)
                    # if(nst>=5):
                        # action = np.argmax(Q[state_to_index(state)])
                    # else:
                        # best_actions = np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                        # if(len(best_actions)>1):
                            # try:
                                # action = np.random.choice(list(set(state_action[state_to_index(state)]).intersection(set(best_actions))))
                            # except:
                                # action = np.random.choice(list(state_action[state_to_index(state)]))
                        # else:
                            # action=best_actions[0]
                    next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                    next_state = index_to_state(next_state_index)
                    reward = rewards[next_state[0], next_state[1]]
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
                if state in goal_states:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
                    # Reset to the initial state if the agent reaches a terminal state
                    state = start_state
        
                action = epsilon_greedy_policy(state, epsilon)
                next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                next_state = index_to_state(next_state_index)
                reward = rewards[next_state[0], next_state[1]]
        
                # Update Q-value
                Q[state_to_index(state), action] += learning_rate * (
                    reward + discount_factor * np.max(Q[state_to_index(next_state)]) - Q[state_to_index(state), action]
                )
        
                state = next_state
                step += 1

                if step % test_steps == 0:
                    tr = test_q()
                    # for k in tr:
                    episode_rewards.append(tr)
                    
            return episode_rewards

        # Run multiple episodes and average results
        num_runs = 1
        N_steps=18000
        test_steps = 12
        num_episodes = int(N_steps/test_steps)
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs,num_episodes))
        for run in range(num_runs):
            Q = np.zeros((S,A))
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)

        average_rewards /= num_runs
        end_time = time.time()
        