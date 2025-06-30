import os
import numpy as np
# np.random.seed(2)
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import time
os.chdir(os.getcwd())
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Load configuration
st = cfg["start"]
ed = cfg["end"]
nst_range = cfg["nst_range"]

gamma = cfg["gamma"]
learning_rate = cfg["learning_rate"]
epsilon_initial = cfg["epsilon_initial"]
epsilon_decay = cfg["epsilon_decay"]
epsilon_min = cfg["epsilon_min"]
max_steps = cfg["max_steps"]
N_steps = cfg["N_steps"]
test_steps = cfg["test_steps"]

x_percentage = cfg["x_percentage"]
bound_update_freq = cfg.get("bound_update_freq", 5000)


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
    terminal_state=eval((f.readline().rstrip()))

    f.close()

    return S, A, R, R2, T, gamma,terminal_state

data=[]
for avg in range(st,ed):
    S, A, R, R2, T, gamma,terminal_state = read_mdp(f"{avg}//mdp_exp_{'01'}.txt")
    for bf in range(1,6,2):
        gamma = 0.9
        x_percentage = 0
        number_stochastic_transitions = bf
        
        T1 = np.load(f"{avg}//T_{bf}.npy")
        
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
        
        np.save(f"{avg}//Q1_{bf}.npy",q_p1)
        np.save(f"{avg}//Q2_{bf}.npy",q_p2)
        
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
        r = R+R2

        Q = q_p1 + q_p2 
        # r=R+R2
        Q[terminal_state,:] = 0#np.max(r)
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
                            temp.append(r[s,a,sdash] + gamma*np.max(Q_k[sdash]))
                        # if(Q_k[s,a]<max(temp)):
                        #     print((i,Q_k[s,a],max(temp)))
                        #     input()
                        Q[s,a] = min(Q_k[s,a],max(temp)) #max(temp)
            # if(i>0):
            #     if(np.max(np.abs(Q_k-Q))> 0.9*(np.max(np.abs(U-Udash)))):
            #         print(i)
            #         input()
            if(np.sum(np.abs(Q-Q_k))<0.0000000001):
                print(i)
                print("------------")
                break
        
        # Q=np.round(Q,2)

        Qm = np.zeros((S,A))
        o1 = q_p1 + q_m2
        o2 = q_p2 + q_m1
        
        for s in range(S):
            for a in range(A):
                Qm[s,a]= max(o1[s,a], o2[s,a])
        
        # Qm +=50
        # rm = R+R2
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
                            temp.append(r[s,a,sdash] + gamma*np.max(Qm_k[sdash]))
                        Qm[s,a] =   max(Qm_k[s,a],min(temp)) #min(temp)
            # if(i>0):
            #     if(np.max(np.abs(Qm_k-Qm))> 0.9*(np.max(np.abs(U-Udash)))):
            #         print("lowerbound")
            #         print(i)
                    # input()
            if(np.sum(np.abs(Qm-Qm_k))<0.0000000001):
                print(i)
                print("------------")
                break
        
        # Qm = np.round(Qm,2)
        
        info=[]
        final_actions=set(list(range(A)))
        prune={}
        state_action={}
        c=0
        
        for i in range(S):
            alist=[]
            for action_l in range(A):
                for action_u in range(A):
                    if(action_l==action_u):
                        continue
                    if( Qm[i, action_l]-Q[i,action_u] >1e-7 ):
                        info.append((i,action_l, action_u))
                        alist.append(action_u)
                        c=c+1
            prune[i]= set(alist)
            state_action[i]= final_actions.difference(set(alist))
        
        print(S*A-sum([len(state_action[i]) for i in state_action.keys()]))
        
        ##############################################################################
        violations1=[]
        violations2=[]
        def epsilon_greedy_policy(state, epsilon):
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
                        violations1.append((avg, state, state_action[state], Q[state].copy(),(action, epsilon)))
                    else:
                        violations2.append((avg, state, state_action[state], Q[state].copy(),(action, epsilon)))
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
                while state not in terminal_state and step<max_steps:
                    step+=1
                    action = epsilon_greedy_policy(state, 0)
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
            state = 0
            step = 1
        
            while step < N_steps+1:
                if state in terminal_state:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    # Reset to the initial state if the agent reaches a terminal state
                    state = 0
        
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
        
        # Run multiple episodes and average results
        num_runs = 1
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
        