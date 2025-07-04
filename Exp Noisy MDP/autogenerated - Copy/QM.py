import os
import numpy as np
# np.random.seed(2)
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import time
os.chdir(os.getcwd())
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
        # print(s)
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
# S, A, R, R2, T, gamma,terminal_state = read_mdp(f"mdp_exp_{'01'}.txt")
# st = int(input("start: "))
# ed = int(input("end: "))
def read_and_increment(filename):
    try:
        # Read the current values from the file
        with open(filename, 'r') as f:
            lines = f.readlines()
            st = int(lines[0].strip())
            ed = int(lines[1].strip())

        # Increment the values
        st += 5
        ed += 5

        # Write the incremented values back to the file
        with open(filename, 'w') as f:
            f.write(str(st) + '\n')
            f.write(str(ed) + '\n')

        return st, ed
    except FileNotFoundError:
        print("File not found. Creating a new file with default values.")
        with open(filename, 'w') as f:
            f.write('0\n')
            f.write('5\n')
        return 0, 5

# Example usage
filename = "RA_idx.txt"
st = int(input("start: "))
ed = int(input("end: "))
print("New values:", st, ed)
for avg in range(st,ed):

    S, A, R, R2, T, gamma,terminal_state = read_mdp(f"{avg}//mdp_exp_{'01'}.txt")
    bf=1
    number_stochastic_transitions=bf
    gamma = 0.9
    x_percentage = 0
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
    
    
    
    for noise in [0,0.015,0.03]:
        # r = np.load(f"{avg}//r_target_{noise}.npy")
        #r=R + R2 + np.random.uniform(-noise,noise,size=R.shape)
        #np.save(f"{avg}//r_target_{noise}.npy", r)
        np.load(f"{avg}//r_target_{noise}.npy")
        approx_r = r +noise
        approx_rm= r-noise
        # Flatten the arrays
        R_flat = R.flatten()
        R2_flat = R2.flatten()
        R_target_flat = r.flatten()
        
        # Stack R and R2 horizontally
        X = np.column_stack((R_flat, R2_flat))
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, R_target_flat)
        
        # Get coefficients
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Predict R_target
        R_target_pred = model.predict(X)
        
        # Compute mean squared error
        mse = mean_squared_error(R_target_flat, R_target_pred)
        print(mse)
        
        def transition(current_state, action):
            global T1, S, number_stochastic_transitions
            p=T1[current_state, action]
            t=np.argsort(p)[-number_stochastic_transitions:]
            return t[np.where(T1[current_state,action,np.argsort(T1[current_state,action])[-number_stochastic_transitions:]]!=0)]
        
       
        start_time = time.time()

        Q = coefficients[0]*q_p1 + coefficients[1]*q_p2+ intercept +(9.08*noise)
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
                            temp.append(approx_r[s,a,sdash] + gamma*np.max(Q_k[sdash]))
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
        
        Qm = np.zeros((S,A))
        o1 = coefficients[0]*q_p1 + coefficients[1]*q_m2 + intercept 
        o2 = coefficients[0]*q_p2 + coefficients[0]*q_m1 + intercept 
    
        for s in range(S):
            for a in range(A):
                Qm[s,a]= max(o1[s,a], o2[s,a]) +(9.08*(-noise))
        
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
                            temp.append(approx_rm[s,a,sdash] + gamma*np.max(Qm_k[sdash]))
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
        r=R+R2
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
        
        def test_q(e=30):
            global Q
            episode_rewards=[]
            for episode in range(e):
                state = 0
                total_reward = 0
                step = 0
                while state not in terminal_state and step<max_steps:
                    step+=1
                    # action = np.argmax(Q[state])
                    best_actions = np.where(Q[state] == np.max(Q[state]))[0]
                    if(len(best_actions)>1):
                        try:
                            action = np.random.choice(list(set(state_action[state]).intersection(set(best_actions))))
                        except:
                            action = np.random.choice(list(state_action[state]))
                    else:
                        action = best_actions[0]
                    next_state = np.random.choice(S, p=T1[state, action, :])
                    reward = r[state, action, next_state]
                    total_reward += reward
                    state = next_state
                episode_rewards.append(total_reward)
            return np.mean(episode_rewards)#total_reward 
        
        def q_learning(N_steps, test_steps):
            global Q
            epsilon = epsilon_initial
            episode_rewards = []
            state = 0
            step = 1
            e_step=0
            while step < N_steps+1:
                e_step+=1
                if state in terminal_state or e_step%max_steps==0:
                    e_step=0
                    # Decay epsilon
                    # epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    # Reset to the initial state if the agent reaches a terminal state
                    state = 0
        
                action = epsilon_greedy_policy(state, epsilon)
                epsilon = max(epsilon * epsilon_decay, epsilon_min)
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
        epsilon_decay = 0.999988#0.999979#0.997
        epsilon_min = 0.01
        # num_episodes = 4000
        max_steps = 60 #S
        
        N_steps=400000#1200000
        test_steps = 40#50#100#S*5 #10
        # N_steps=30000
        # test_steps = 6 #12
        num_episodes = int(N_steps/test_steps)
        
        # Run multiple episodes and average results
        num_runs = 1
        average_rewards = np.zeros(num_episodes)
        rewards_run = np.zeros((num_runs, num_episodes))
        for run in range(num_runs):
            Q= np.zeros((S,A))
            # np.random.seed(run)
            episode_rewards = q_learning(N_steps,test_steps)
            rewards_run[run] = episode_rewards
            average_rewards += np.array(episode_rewards)
        average_rewards /= num_runs
        end_time = time.time()
        pd.DataFrame(rewards_run).to_csv(f'{avg}//ours_{x_percentage}_{noise}.csv')
        w=100
        # Plot average Q value per episode over 5 runs
        plt.plot(range(num_episodes-w+1), np.convolve(average_rewards, np.ones(w), 'valid') / w)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title(f'Average Reward per Episode over {num_runs} Runs')
        plt.show()

        pd.DataFrame(rewards_run).to_csv(f'{avg}//ours_{x_percentage}_{noise}.csv')

        data.append((avg, f"ours_{noise}", S, A, S*A-sum([len(state_action[i]) for i in state_action.keys()])))
        # break
pd.DataFrame(data, columns=['Run', 'Domain', '|S|', '|A|', 'Actions Pruned']).to_csv(f"Data_RA_{ed-1}.csv")

