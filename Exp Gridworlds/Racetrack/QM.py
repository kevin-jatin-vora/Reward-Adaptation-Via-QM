import numpy as np
#np.random.seed(6)
from matplotlib import pyplot as plt
from scipy import stats
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
#######################################################################################################################


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
            T_stochastic[s,a] = T1[s,a]
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
data=[]
for avg in range(init,last):
    for nst in nst_range:
        x_percentage = 0
        number_stochastic_transitions = nst #A
        gamma=gamma

        
        filename = f'{avg}//T_{nst}.npy'
        T1=np.load(filename)
        
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
        q_p1 = compute_q_values(S, A, R, T1, gamma)
        
        # Compute Q-values
        q_p2 = compute_q_values(S, A, R2, T1, gamma)
        
        q_p3 = compute_q_values(S, A, R3, T1, gamma)
        
        def compute_q_values_mu(S, A, R, T, gamma):
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
        q_m1 = compute_q_values_mu(S, A, R, T1, gamma)
        # Compute Q-values
        q_m2 = compute_q_values_mu(S, A, R2, T1, gamma)
        
        q_m3 = compute_q_values_mu(S, A, R3, T1, gamma)
        
        # Qstar = compute_q_values(S, A, R+R2+R3, T1, gamma)
        
        def transition_neighbours(current_state, action):
            global T1, S, number_stochastic_transitions
            p=T1[current_state, action]
            # if(np.sum(p)==1):
            t=np.argsort(p)[-number_stochastic_transitions:]
            return t[np.where(T1[current_state,action,np.argsort(T1[current_state,action])[-number_stochastic_transitions:]]!=0)]
        
        start_time = time.time()
        r = R+R2+R3
        
        Q = q_p1 + q_p2 + q_p3
        # r=R+R2+R3
        Q[GOALS,:] = 0
        Q[WALLS,:] = 0
        
        for i in range(5000):
            if(i>0):
                U = Q_k.copy()
                Udash = Q.copy()
            Q_k=Q.copy()
            for s in range(S):
                if(s in GOALS or s in WALLS):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition_neighbours(s,a):
                            temp.append(r[s,a,sdash] + gamma*np.max(Q_k[sdash]))
                        Q[s,a] =min(Q_k[s,a],max(temp)) #max(temp)             
            # if(i>0):
            #     if(np.round(np.max(np.abs(Q_k-Q)),7)> np.round(gamma*(np.max(np.abs(U-Udash))),7)):
            #         # print(np.max(np.abs(Q_k-Q)))
            #         # print(gamma*(np.max(np.abs(U-Udash))))
            #         print(i)
                    # input()
            if(np.sum(np.abs(Q-Q_k))<0.0000000001):
                print(i)
                print("------------")
                break
        
        rm=r
        
        Qm = np.zeros((S,A))
        o1 = q_p1 + q_m2 + q_m3
        o2 = q_p2 + q_m1 + q_m3
        o3 = q_p3 + q_m1 + q_m2
        
        for s in range(S):
            for a in range(A):
                Qm[s,a]= max(o1[s,a], o2[s,a], o3[s,a])#-1
        
        
        Qm[GOALS,:] = 0
        Qm[WALLS,:] = 0
        for i in range(5000):
            if(i>0):
                U = Qm_k.copy()
                Udash = Qm.copy()
            Qm_k=Qm.copy()
            for s in range(S):
                if(s in GOALS or s in WALLS):
                    continue
                else:
                    for a in range(A):
                        temp=[]
                        for sdash in transition_neighbours(s,a):
                            temp.append(rm[s,a,sdash] + gamma*np.max(Qm_k[sdash]))
                        Qm[s,a] = max(Qm_k[s,a],min(temp)) #min(temp)
            # if(i>0):
            #     if(np.max(np.abs(Qm_k-Qm))> gamma*(np.max(np.abs(U-Udash)))):
            #         print("lowerbound")
            #         print(i)
            #         #input()
            if(np.sum(np.abs(Qm-Qm_k))<0.0000000001):
                print(i)
                print("------------")
                break
        
        Q=np.round(Q,4)
        Qm=np.round(Qm,4)
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
                    if( Qm[i, action_l]-Q[i,action_u] > 1e-10 ):
                        info.append((i,action_l, action_u))
                        alist.append(action_u)
            prune[i]= set(alist)
            state_action[i]= final_actions.difference(set(alist))
            
        print(S*A-sum([len(state_action[i]) for i in state_action.keys()]))
        
        ###############################################################################################
        r = R+R2+R3
        
        # Epsilon-greedy policy
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
        # def epsilon_greedy_policy(state, epsilon):
        #     if np.random.rand() < epsilon:
        #         # Exploration: Choose a random action from the available set
        #         return np.random.choice(list(state_action[state]))
        #     else:
        #         # Exploitation: Choose a best action from the available set, breaking ties randomly
        #         state_index = state
        #         q_values = Q[state_index, list(state_action[state_index])]
        #         best_action_indices = np.where(q_values == np.max(q_values))[0]
        #         best_action_index = np.random.choice(best_action_indices)
        #         bas = list(state_action[state_index])
        #         best_action = bas[best_action_index]
        #         return best_action
        
        # def epsilon_greedy_policy(state, epsilon):
        #     if np.random.rand() < epsilon:
        #         return np.random.choice(list(state_action[state]))
        #     else:
        #         best_actions = np.where(Q[state] == np.max(Q[state]))[0]
        #         if(len(best_actions)>1):
        #             try:
        #                 action = np.random.choice(list(set(state_action[state]).intersection(set(best_actions))))
        #             except:
        #                 action = np.random.choice(list(state_action[state]))
        #         else:
        #             action=best_actions[0]
        #         return action
        #         # return np.argmax(Q[state])
        
        def test_q(e=30):
            global Q
            episode_rewards=[]
            for episode in range(e):
                state = START
                total_reward = 0
                step = 0
                while state not in WALLS and state not in GOALS and step<max_steps:
                    step+=1
                    action = epsilon_greedy_policy(state, 0) #np.argmax(Q[state])
                    # best_actions = np.where(Q[state] == np.max(Q[state]))[0]
                    # if(len(best_actions)>1):
                    #     try:
                    #         action = np.random.choice(list(set(state_action[state]).intersection(set(best_actions))))
                    #     except:
                    #         action = np.random.choice(list(state_action[state]))
                    # else:
                    #     action = best_actions[0]
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
        learning_rate = 0.1
        discount_factor = gamma
        epsilon_initial = 1.0
        epsilon_decay = 0.998
        # if(nst>3):
        #     epsilon_decay=0.996
        epsilon_min = 0.01
        # num_episodes = 4000
        max_steps=30 #13
        
        N_steps=28000
        test_steps = 4 #12
        # N_steps=30000
        # test_steps = 6 #12
        num_episodes = int(N_steps/test_steps)
        
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):
            Q = np.ones((S,A))*-0.5  #np.zeros((S,A)) #
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps, test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        average_rewards /= num_runs
        end_time = time.time()
        import pandas as pd
        # pd.DataFrame(rewards_run).to_csv(f'ours_{x_percentage}_{number_stochastic_transitions}.csv')
        # w=100
        # # Plot average Q value per episode over 5 runs
        # plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
        # plt.xlabel('Episode')
        # plt.ylabel('Average Reward')
        # plt.title(f'Average Reward per Episode over {num_runs} Runs')
        # plt.show()

        pd.DataFrame(rewards_run).to_csv(f'{avg}//ours_{x_percentage}_{number_stochastic_transitions}.csv')

        
        heat_map=np.zeros((7,7))    
        for key in state_action.keys():
            heat_map[STATE2WORLD[key]]=len(state_action[key])
        
        import seaborn as sns
        cmap =  'Blues' #sns.cm.flare
        ax = sns.heatmap(heat_map, linewidth=0.5, linecolor='black', cmap=cmap, alpha=0.6)
        #ax = sns.heatmap(heat_map, linewidth=0.5, cmap=cmap, alpha=0.6)
        #ax.invert_yaxis()
        #ax.invert_xaxis()
        plt.savefig(f"{avg}//heatmap_RT_{x_percentage}_{number_stochastic_transitions}.png",bbox_inches = 'tight', dpi=1000)
        plt.show()
        data.append((avg, f"RT_{x_percentage}_{number_stochastic_transitions}", S, A, S*A-sum([len(state_action[i]) for i in state_action.keys()]), end_time-start_time))
pd.DataFrame(data, columns=['Run', 'Domain', '|S|', '|A|', 'Actions Pruned', 'QM']).to_csv(f"Data_RA_{last-1}.csv")
f = open("test_ep.txt", "w")
f.write(f"{test_steps}")
f.close()