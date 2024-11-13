import numpy as np
#np.random.seed(6)
from matplotlib import pyplot as plt
from scipy import stats
import time
import os
np.random.seed(2)
import pandas as pd

os.chdir(os.getcwd())
S=49
A=7
START = 42
GOALS = [0, 7]  # state index of goals
WALLS = [  # state index of walls
    5, 6,
    14, 15,
    21, 22, 23, 24, 25,
    28, 29,
    46, 47, 48
]
STATE2WORLD = {
    0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6),
    7: (1, 0), 8: (1, 1), 9: (1, 2), 10: (1, 3), 11: (1, 4), 12: (1, 5), 13: (1, 6),
    14: (2, 0), 15: (2, 1), 16: (2, 2), 17: (2, 3), 18: (2, 4), 19: (2, 5), 20: (2, 6),
    21: (3, 0), 22: (3, 1), 23: (3, 2), 24: (3, 3), 25: (3, 4), 26: (3, 5), 27: (3, 6),
    28: (4, 0), 29: (4, 1), 30: (4, 2), 31: (4, 3), 32: (4, 4), 33: (4, 5), 34: (4, 6),
    35: (5, 0), 36: (5, 1), 37: (5, 2), 38: (5, 3), 39: (5, 4), 40: (5, 5), 41: (5, 6),
    42: (6, 0), 43: (6, 1), 44: (6, 2), 45: (6, 3), 46: (6, 4), 47: (6, 5), 48: (6, 6)
}
R1 = np.load("R1.npy")
R2 = np.load("R2.npy")
R3 = np.load("R3.npy")
r=R1+R2+R3
gamma=0.88
data=[]
init = int(input("init: "))
last = int(input("end: "))
for avg in range(init,last):
    for nst in range(1,8,2):
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
        q_p1 = compute_q_values(S, A, R1, T1, gamma)
        
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
        q_m1 = compute_q_values_mu(S, A, R1, T1, gamma)
        # Compute Q-values
        q_m2 = compute_q_values_mu(S, A, R2, T1, gamma)
        
        q_m3 = compute_q_values_mu(S, A, R3, T1, gamma)
        
        # Qstar = compute_q_values(S, A, R+R2+R3, T1, gamma)
        
        def transition_neighbours(current_state, action):
            global T1, nst
            p=T1[current_state, action]
            # if(np.sum(p)==1):
            t=np.argsort(p)[-nst-1:]
            return t[np.where(T1[current_state,action,np.argsort(T1[current_state,action])[-nst-1:]]!=0)]
        
        start_time = time.time()
        
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
                            temp.append(r[s,a,sdash] + gamma*np.max(Qm_k[sdash]))
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
        
        def epsilon_greedy_policy(state, epsilon):
            if np.random.rand() < epsilon:
                return np.random.choice(list(state_action[state]))
            else:
                best_actions = np.where(Q[state] == np.max(Q[state]))[0]
                if(len(best_actions)>1):
                    try:
                        action = np.random.choice(list(set(state_action[state]).intersection(set(best_actions))))
                    except:
                        action = np.random.choice(list(state_action[state]))
                else:
                    action=best_actions[0]
                return action
                # return np.argmax(Q[state])
        
        def test_q(e=30):
            global Q
            episode_rewards=[]
            for episode in range(e):
                state = START
                total_reward = 0
                step = 0
                while state not in WALLS and state not in GOALS and step<max_steps:
                    step+=1
                    action = epsilon_greedy_policy(state,0) 
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
        epsilon_min = 0.01
        max_steps=30 #13
        
        N_steps=28000
        test_steps = 4 #12
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
        
        w=100
        # Plot average Q value per episode over 5 runs
        plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title(f'Average Reward per Episode over {num_runs} Runs')
        plt.show()

        pd.DataFrame(rewards_run).to_csv(f'{avg}//ours_{nst+1}.csv')

        
        heat_map=np.zeros((7,7))    
        for key in state_action.keys():
            heat_map[STATE2WORLD[key]]=len(state_action[key])
        
        import seaborn as sns
        cmap =  'Blues' #sns.cm.flare
        ax = sns.heatmap(heat_map, linewidth=0.5, linecolor='black', cmap=cmap, alpha=0.6)
        #ax = sns.heatmap(heat_map, linewidth=0.5, cmap=cmap, alpha=0.6)
        #ax.invert_yaxis()
        #ax.invert_xaxis()
        plt.savefig(f"{avg}//heatmap_RT_{nst+1}.png",bbox_inches = 'tight', dpi=1000)
        plt.show()
        data.append((avg, f"RT_{nst+1}", S, A, S*A-sum([len(state_action[i]) for i in state_action.keys()]), end_time-start_time))

pd.DataFrame(data, columns=['Run', 'Domain', '|S|', '|A|', 'Actions Pruned', 'QM']).to_csv(f"Data_RA_{last-1}.csv")
f = open("test_ep.txt", "w")
f.write(f"{test_steps}")
f.close()