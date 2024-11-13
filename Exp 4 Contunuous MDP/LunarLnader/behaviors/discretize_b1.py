import numpy as np
import gym
env = gym.make("LunarLander-v2")
np.random.seed(2)

def set_buckets_and_actions():
    number_of_buckets =  (5,5,5,5,5,5,1,1)#buckets in each dimension
    number_of_actions = env.action_space.n
    
    #Creating a 2-tuple with the original bounds of each dimension
    state_value_bounds = list(zip(env.observation_space.low,env.observation_space.high))
    
    return number_of_buckets, number_of_actions, state_value_bounds

number_of_buckets, number_of_actions, state_value_bounds = set_buckets_and_actions()


def bucketize(state):
    bucket_indexes = []
    for i in range(len(state)):
        if state[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state[i] >= state_value_bounds[i][1]:
            bucket_index = number_of_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (number_of_buckets[i]-1) * state_value_bounds[i][0]/bound_width
            scaling = (number_of_buckets[i]-1) / bound_width
            bucket_index = int((scaling*state[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)

def de_bucketize(bucket_indexes):
    state = []
    for i in range(len(bucket_indexes)):
        if(i>5):
            state.append(0.5)
            continue
        # Calculate the range of values for the current bucket
        bucket_min = state_value_bounds[i][0] + (bucket_indexes[i] / (number_of_buckets[i] - 1)) * (state_value_bounds[i][1] - state_value_bounds[i][0])
        bucket_max = state_value_bounds[i][0] + ((bucket_indexes[i] + 1) / (number_of_buckets[i] - 1)) * (state_value_bounds[i][1] - state_value_bounds[i][0])
        
        # Calculate the center of the bucket
        bucket_center = (bucket_min + bucket_max) / 2
        
        state.append(bucket_center)
    # Clip last two elements to ensure they are within bounds
    state[-1] = np.clip(np.round(state[-1] + np.random.uniform(-1,1)), 0, 1)
    state[-2] = np.clip(np.round(state[-2] + np.random.uniform(-1,1)), 0, 1)
    return state
        


import itertools
import numpy as np

def iterate_all_states():
    all_states = []
    # Generate all possible combinations of bucket indexes
    bucket_index_combinations = itertools.product(*[range(n) for n in number_of_buckets])
    
    # Iterate through each combination
    for bucket_indexes in bucket_index_combinations:
        # Call de_bucketize twice for each combination to get two states
        state1 = de_bucketize(bucket_indexes)
        # state2 = de_bucketize(bucket_indexes)
        # state3 = de_bucketize(bucket_indexes)
        all_states.append(state1)
        # all_states.append(state2)
        # all_states.append(state3)
    
    return all_states

# Call the function to get all possible states
all_states = iterate_all_states()
unique_states = np.unique(all_states, axis=0)

print("Total number of states:", len(all_states))


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Define QNetwork class
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# Load the DQN model
state_size = len(all_states[0])  # Dimension of each state
action_size = number_of_actions
seed = 0  # Random seed
qnetwork_local = QNetwork(state_size, action_size, seed)
qnetwork_local.load_state_dict(torch.load('checkpoint_b1.pth'))

# Create a Q-table
Q_table = defaultdict(float)
from tqdm import tqdm
Q_sum_per_bucket = defaultdict(lambda: [0.0] * action_size)
state_count_per_bucket = defaultdict(int)

# Predict Q-values for all states
with tqdm(total=len(all_states)) as pbar:
    for state in all_states:
        # Bucketize the state
        bucketized_state = bucketize(state)
        bucketized_state = tuple(bucketized_state)  # Convert to tuple for dictionary key
        
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Predict Q-values using the loaded DQN model
        with torch.no_grad():
            q_values = qnetwork_local(state_tensor)
        
        # Accumulate Q-values for the corresponding bucket index
        Q_sum_per_bucket[bucketized_state] += q_values.numpy()[0]
        
        # Increment state count for the corresponding bucket index
        state_count_per_bucket[bucketized_state] += 1
        
        # Update progress bar
        pbar.update(1)

# Calculate average Q-values for each bucket index
for bucketized_state in Q_sum_per_bucket:
    Q_sum_per_bucket[bucketized_state] /= state_count_per_bucket[bucketized_state]

# Assign the averaged Q-values to the Q_table
Q_table = Q_sum_per_bucket


print("Q-table created successfully!")
import dill

# Save Q-table using dill
with open('LL_b1.pkl', 'wb') as f:
    dill.dump(Q_table, f)

print("Q-table saved successfully!")