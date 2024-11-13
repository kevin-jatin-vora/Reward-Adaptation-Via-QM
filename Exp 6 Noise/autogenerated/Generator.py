import numpy as np
import random
import os
os.chdir(os.getcwd())

num_actions = 8#random.randint(9, 12) #8
num_states = 50#random.randint(40, 50) #50
print(num_actions)
print(num_states)

# Generate Transition probabilities for each state-action pair
transitions = []
# Generate Rewards for each state-action pair
rewards = np.ones((num_states, num_actions, num_states))*-0.001
rewards2 = np.ones((num_states, num_actions, num_states))*-0.001

for i in range(num_states):
    row = []
    for j in range(num_actions):
        action_probs = [0.0] * num_states  # Initialize all transition probabilities to 0
        
        # Randomly select `num_actions - 1` states in addition to the high_prob_state
        possible_states = np.random.choice(num_states, num_actions, replace=False)
        high_prob_state = possible_states[0]  # Assign one state the high probability
        other_states = possible_states[1:]    # The rest will get a portion of 0.3
        rewards[i,j,high_prob_state] = np.random.uniform(-5,-1)
        rewards2[i,j,high_prob_state] = np.random.uniform(1,5)
        action_probs[high_prob_state] = 0.7  # Assign 0.7 probability to the selected high-prob state
        
        # Distribute the remaining 0.3 among the other states
        if len(other_states) > 0:
            remaining_probs = list(np.random.dirichlet(np.ones(len(other_states))) * 0.3)
            for idx, state in enumerate(other_states):
                action_probs[state] = remaining_probs[idx]
        
        row.append(action_probs)
    transitions.append(row)

terminal_state = list(np.random.randint(low=1,high=num_states, size=3))
# print("Terminal state:", terminal_state)
# transitions = np.array(transitions)

# x=[]
# y=[]
# for i in terminal_state:
#     x.append(np.where(transitions[:,:,i]==np.max(transitions[:,:,i]))[0][0])
#     y.append(np.where(transitions[:,:,i]==np.max(transitions[:,:,i]))[1][0])
# # Generate Rewards for each state-action pair
# rewards = np.ones((num_states, num_actions, num_states))*-0.001
# rewards2 = np.ones((num_states, num_actions, num_states))*-0.001
# # rewards3 = np.zeros((num_states, num_actions, num_states))


# # Update rewards for the selected states
# # State 1: r1 = +1, r2 = -1, r3 = 0
# rewards[x[0],y[0],terminal_state[0]] = 1
# rewards2[x[0],y[0],terminal_state[0]] = -1

# # State 2: r1 = -1, r2 = +1, r3 = 0
# rewards[x[1],y[1],terminal_state[1]] = -1
# rewards2[x[1],y[1],terminal_state[1]] = 1

# # State 3 (Goal): r1 = 0.6, r2 = 0.6, r3 = 1.2
# rewards[x[2],y[2],terminal_state[2]] = 0.6
# rewards2[x[2],y[2],terminal_state[2]] = 0.6

# Convert to lists for writing to file
# transitions = transitions.tolist()
rewards = rewards.tolist()
rewards2 = rewards2.tolist()

# Write MDP to file
with open('mdp_exp_01.txt', 'w') as f:
    f.write(f'{num_states}\n{num_actions}\n')
    for row in rewards:
        for action_probs in row:
            f.write('\t'.join(str(val) for val in action_probs) + '\n')
    for row in rewards2:
        for action_probs in row:
            f.write('\t'.join(str(val) for val in action_probs) + '\n')
    for row in transitions:
        for action_probs in row:
            f.write('\t'.join(str(val) for val in action_probs) + '\n')
    f.write('0.99\n')
    f.write(f'{terminal_state}\n')
