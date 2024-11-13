import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import time

os.chdir(os.getcwd())
S=45
A=4
R1 = np.load("0//R1.npy")
R2 = np.load("0//R2.npy")
r=R1+R2
terminal_state = np.load("terminal.npy")
gamma = 0.939
data=[]
last=3
for avg in range(0,3):
    for nst in range(1,5):
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
        
        # Assuming read_mdp function is defined as mentioned in the question
        #S, A, R, R2, T, gamma, terminal_state = read_mdp("mdp_exp1.txt")
        
        # Compute Q-values
        q_p1 = compute_q_values(S, A, R1, T1, gamma)
        
        # Compute Q-values
        q_p2 = compute_q_values(S, A, R2, T1, gamma)
        
        # Qstar = compute_q_values(S, A, R+R2, T1, gamma, terminal_state)
        
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
        
        
        def transition(current_state, action):
            global T1, S, number_stochastic_transitions
            p=T1[current_state, action]
            # if(np.sum(p)==1):
            t=np.argsort(p)[-nst-1:]
            return t[np.where(T1[current_state,action,np.argsort(T1[current_state,action])[-nst-1:]]!=0)]
            # else:
            #     return []
        
        start_time = time.time()
        
        real_r = R1+R2
        
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
            # if(i>0):
            #     if(np.round(np.max(np.abs(Q_k-Q)),7)> np.round(gamma*(np.max(np.abs(U-Udash))),7)):
            #         # print(np.max(np.abs(Q_k-Q)))
            #         # print(gamma*(np.max(np.abs(U-Udash))))
            #         print(i)
            #         # input()
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
            # if(i>0):
            #     if(np.max(np.abs(Qm_k-Qm))> gamma*(np.max(np.abs(U-Udash)))):
            #         # print("lowerbound")
            #         print(i)
            #         # input()
            if(np.max(np.abs(Qm-Qm_k))<0.0000000001):
                print(i)
                print("------------")
                break
        
        # Qm = np.round(Qm,2)
        
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
            
        # print(S*A-sum([len(state_action[i]) for i in state_action.keys()]))
        
        # input()
        ########################################################################################
        # rr=R+R2
    
        # Define environment parameters
        num_rows = 5
        num_cols = 9
        num_states = num_rows * num_cols
        num_actions = 4
        start_state = (0,4)
        goal_states = terminal_state#[(0, 0), (0, 8), (4, 4)]
        
        # Define rewards
        # rewards = np.ones((num_rows, num_cols))*-0.0001
        # rewards[0, 0] = 1.0
        # rewards[0, 8] = 1.0
        # rewards[4, 4] = 1.2
        
        # Define Q-learning parameters
        learning_rate = 0.1
        discount_factor = gamma #0.95
        epsilon_initial = 1.0
        epsilon_decay = 0.9995 #[0.999,0.999, 0.9995, 0.999]
        epsilon_min = 0.01
        
        max_steps=30
        # Define actions
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
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
                return np.random.choice(list(state_action[state_to_index(state)]))
            else:
                # Get indices of maximum Q-values
                best_actions = np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                # Randomly select one of the best actions
                if(len(best_actions)>1):
                    try:
                        return np.random.choice(list(set(state_action[state_to_index(state)]).intersection(set(best_actions))))
                    except:
                        return np.random.choice(list(state_action[state_to_index(state)]))
                else:
                    return np.random.choice(best_actions)
                # return np.argmax(Q[state_to_index(state)])
        
        def test_q(e=30):
            global Q
            episode_rewards = []
            for episode in range(e):
                state = start_state
                total_reward = 0
                step = 0
                while state_to_index(state) not in goal_states and step<max_steps:
                    step+=1
                    # if(nst>=5):
                    #     action = np.argmax(Q[state_to_index(state)])
                    # else:
                    action = epsilon_greedy_policy(state,0)#np.where(Q[state_to_index(state)] == np.max(Q[state_to_index(state)]))[0]
                    # if(len(best_actions)>1):
                    #     try:
                    #         action = np.random.choice(list(set(state_action[state_to_index(state)]).intersection(set(best_actions))))
                    #     except:
                    #         action = np.random.choice(list(state_action[state_to_index(state)]))
                    # else:
                    #     action=best_actions[0]
                    next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                    next_state = index_to_state(next_state_index)
                    reward = r[state_to_index(state), action, next_state_index]
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
                if state_to_index(state) in goal_states:
                    # Decay epsilon
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
            
                    # Reset to the initial state if the agent reaches a terminal state
                    state = start_state
        
                action = epsilon_greedy_policy(state, epsilon)
                next_state_index = np.random.choice(num_states, p=T1[state_to_index(state), action, :])
                next_state = index_to_state(next_state_index)
                reward = r[state_to_index(state), action, next_state_index]
        
                # Update Q-value
                Q[state_to_index(state), action] += learning_rate * (
                    reward + discount_factor * np.max(Q[state_to_index(next_state)]) - Q[state_to_index(state), action]
                )
        
                state = next_state
                step += 1
        
                # Track reward for each step
                # episode_rewards.append(reward)
        
                # Decay epsilon
                # epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
                # Optional: Print or test the Q-values periodically
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
        pd.DataFrame(rewards_run).to_csv(f'{avg}//ours_{nst+1}.csv')
        
        w=30
        # Plot average Q value per episode over 5 runs
        plt.plot(np.arange(num_episodes-w+1)*test_steps, np.convolve(average_rewards, np.ones(w), 'valid') / w)
        plt.xlabel('step')
        plt.ylabel('Average Return')
        # plt.title(f'Average Reward per 3000 steps over {num_runs} Runs')
        plt.show()
        # input()
        
        xy_s={}
        for i in range(5):
            for j in range(9):
                xy_s[i*9 + j]=(i,j) 
        
        heat_map=np.zeros((5,9))    
        for key in state_action.keys():
            heat_map[xy_s[key]]=len(state_action[key])
        
        
        cmap =  'Blues' #sns.cm.flare
        ax = sns.heatmap(heat_map, linewidth=0.5, linecolor='black', cmap=cmap, alpha=0.6)
        # ax.invert_yaxis()
        plt.savefig(f"{avg}//heatmap_DE_{nst+1}.png",bbox_inches = 'tight', dpi=1000)
        plt.show()
        # input()
        data.append((avg, f"Dollar-Euro_{nst+1}", S, A, S*A-sum([len(state_action[i]) for i in state_action.keys()]), end_time-start_time))
        
        # break
    # break
        
pd.DataFrame(data, columns=['Run', 'Domain', '|S|', '|A|', 'Actions Pruned', 'QM']).to_csv(f"Data_RA_{last-1}.csv")
f = open("test_ep.txt", "w")
f.write(f"{test_steps}")
f.close()