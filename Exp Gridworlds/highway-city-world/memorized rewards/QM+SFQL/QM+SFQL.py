import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR


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

def epsilon_greedy_is(Q_star, Q_mu, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, A)  # Random action
    else:
        # Select one action from Q_star using argmax and another from Q_mu using argmin
        action_star = np.argmax(Q_star[state])  # Action that maximizes Q_star
        action_mu = np.argmin(Q_mu[state])  # Action that minimizes Q_mu
        
        # Randomly return one of these two actions
        return np.random.choice([action_star, action_mu], p=[0.5,0.5])
    
def episodic_q_learning(S, A, T1, R, start_state, goals):
    alpha = 0.1
    gamma = 0.9
    epsilon_initial = 1.0
    epsilon_decay = 0.998 #0.992
    epsilon_min = 0.01
    max_steps = S
    N_steps=15000
    # test_steps = 2 #12
    # num_episodes = int(N_steps/test_steps)
    r_indv = np.ones((S,A,S))*-1
    Q_star = np.zeros((S, A))
    Q_mu = np.zeros((S, A))    

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
        action = epsilon_greedy_is(Q_star, Q_mu, state, epsilon)
        next_state = np.random.choice(range(S), p=T[state, action])
        reward = R[state, action, next_state]
        r_indv[state, action, next_state] = R[state, action, next_state]
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

    return Q_star, Q_mu, r_indv
  

def compute_q_values(S, A, R, T, gamma):
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


# Compute Q-values
# q_p1_old = compute_q_values(S, A, R1, T, gamma)
q_p1, q_m1, r1 = episodic_q_learning(S, A, T, R1, start_state, terminal_state)
# Compute Q-values
# q_p2_old = compute_q_values(S, A, R2, T, gamma)
q_p2, q_m2, r2 = episodic_q_learning(S, A, T, R2, start_state, terminal_state)

# def compute_q_values_mu(S, A, R, T, gamma, terminal_state):
#     # Initialize Q-values to zeros
#     Q_new = np.zeros((S, A))
    
#     # Maximum number of iterations for value iteration
#     max_iterations = 5000
    
#     # Value iteration
#     for _ in range(max_iterations):
#         Q = Q_new.copy()
#         for s in range(S):
#             for a in range(A):
#                 q_sa = 0
#                 for s_prime in range(S):
#                     q_sa += T[s][a][s_prime] * (R[s][a][s_prime] + gamma * np.min(Q[s_prime]))
#                 Q_new[s][a] = q_sa
#         if np.max(np.abs(Q - Q_new)) < 1e-12:  # Check for convergence
#             print("Converged in", _ + 1, "iterations")
#             break
#         Q = Q_new
    
#     return Q

# # Compute Q-values
# q_p1 = compute_q_values(S, A, R1, T, gamma)
# # Compute Q-values
# q_p2 = compute_q_values(S, A, R2, T, gamma)
# # Compute Q-values
# q_m1 = compute_q_values_mu(S, A, R1, T, gamma, terminal_state)
# # Compute Q-values
# q_m2 = compute_q_values_mu(S, A, R2, T, gamma, terminal_state)

def policy_evaluation(Q1, transition_probabilities, rewards, discount_factor=gamma, theta=1e-9):
    Q1_eval = np.copy(Q1)
    while True:
        delta = 0
        for state in range(Q1.shape[0]):
            action = np.argmax(Q1[state])
            next_state_probs = transition_probabilities[state, action]
            next_state_rewards = rewards[state, action]
            Q1_new = np.sum(next_state_probs * (next_state_rewards + discount_factor * np.max(Q1_eval, axis=1)))
            delta = max(delta, np.abs(Q1_eval[state, action] - Q1_new))
            Q1_eval[state, action] = Q1_new
        if delta < theta:
            break
    return Q1_eval

Q1_e = policy_evaluation(q_p1_old, T, r)
Q2_e = policy_evaluation(q_p2_old, T, r)

# Initialize combined Q-table
combined_Q = np.zeros((S, A))
# Iterate over each state-action pair
for s in range(A):
    for a in range(A):
        # Find maximum Q-value across both Q-tables for state s and action a
        max_Q_value = max(Q1_e[s, a], Q2_e[s, a])
        
        # Assign the maximum Q-value to the combined Q-table
        combined_Q[s, a] = max_Q_value

bf=2
def transition(current_state, action):
    global T, S, bf
    p=T[current_state, action]
    t=np.argsort(p)[-bf:]
    return t[np.where(T[current_state,action,np.argsort(T[current_state,action])[-bf:]]!=0)]

   
start_time = time.time()

Q = q_p1 + q_p2 +1.4
# Q = np.load("QM+SFQL_UB_init.npy")
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
                    temp.append(r1[s,a,sdash]+r2[s,a,sdash] + gamma*np.max(Q_k[sdash]))
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
        Qm[s,a]= max(o1[s,a], o2[s,a])-1.4

# Qm = np.load("QM+SFQL_LB_init.npy")
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
                    temp.append(r1[s,a,sdash]+r2[s,a,sdash] + gamma*np.max(Qm_k[sdash]))
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


# Qm = np.zeros((S,A))
# o1 = q_p1 + q_m2
# o2 = q_p2 + q_m1

# for s in range(S):
#     for a in range(A):
#         Qm[s,a]= max(o1[s,a], o2[s,a])-1.4
# np.save("memorized rewards//QM+SFQL//QM+SFQL_LB_MR_init.npy", Qm)

# np.save("memorized rewards//QM+SFQL//QM+SFQL_UB_MR_init.npy", q_p1 + q_p2 +1.4)

# np.save("memorized rewards//QM+SFQL//R1_memorized.npy", r1)

# np.save("memorized rewards//QM+SFQL//R2_memorized.npy", r2)
######################################################################################
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        # Exploration: Choose a random action from the available set
        return np.random.choice(list(state_action[state]))
    else:
        # Exploitation: Choose a best action from the available set, breaking ties randomly
        state_index = state
        q_values = Q[state_index, list(state_action[state_index])]
        best_action_indices = np.where(q_values == np.max(q_values))[0]
        best_action_index = np.random.choice(best_action_indices)
        bas = list(state_action[state_index])
        best_action = bas[best_action_index]
        return best_action


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
    Q = combined_Q.copy()
    for k, v in prune.items():
        for at in list(v):
            Q[k,at] = -10
    # np.random.seed(run)
    episode_rewards = q_learning(N_steps, test_steps)
    rewards_run[run] = episode_rewards
    average_rewards += np.array(episode_rewards)
average_rewards /= num_runs
end_time = time.time()
# np.save("memorized rewards//QM+SFQL_ours_learning.npy", rewards_run)
np.save("memorized rewards//QM+SFQL//QM+SFQL_ours_MR_learning.npy", rewards_run)

w=100
# Plot average Q value per episode over 5 runs
plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.title(f'Average Reward per Episode over {num_runs} Runs')
plt.show()
