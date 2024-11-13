import os
import numpy as np
# np.random.seed(2)
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import time
os.chdir(os.getcwd())

data=[]
#S, A, R, R2, T, gamma,terminal_state = read_mdp(f"mdp_exp_{'01'}.txt")
st = int(input("start: "))
ed = int(input("end: "))
for avg in range(st,ed):
    S = np.load(f"{avg}//num_states_16.npy")
    A = np.load(f"{avg}//num_actions_16.npy")
    R = np.load(f"{avg}//R1_16.npy")
    R2 = np.load(f"{avg}//R2_16.npy")
    gamma = 0.9
    terminal_state = np.load(f"{avg}//terminal_states_16.npy")

    for bf in range(1,10,4):
        T1 = np.load(f"{avg}//T_{bf}.npy")
        # T1 = restrict_transition(T1, bf)
        # T1[terminal_state,:, :] = 0
        r=R+R2
        
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
            global T1, S, bf
            p=T1[current_state, action]
            t=np.argsort(p)[-bf:]
            return t[np.where(T1[current_state,action,np.argsort(T1[current_state,action])[-bf:]]!=0)]
        
       
        start_time = time.time()

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
        learning_rate = 0.1
        discount_factor = gamma
        epsilon_initial = 1.0
        epsilon_decay = 0.997
        # if(nst>3):
        #     epsilon_decay=0.996
        epsilon_min = 0.01
        # num_episodes = 4000
        max_steps=S #13
        
        N_steps=20000
        test_steps = 4 #12
        # N_steps=30000
        # test_steps = 6 #12
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
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.title(f'Average Reward per Episode over {num_runs} Runs')
        plt.show()

#         pd.DataFrame(rewards_run).to_csv(f'{avg}//ours_{0}_{bf}.csv')

#         data.append((avg, f"ours_{bf}", S, A, S*A-sum([len(state_action[i]) for i in state_action.keys()]), end_time-start_time))
#         # break
# pd.DataFrame(data, columns=['Run', 'Domain', '|S|', '|A|', 'Actions Pruned', 'QM']).to_csv(f"Data_RA_{ed-1}.csv")

