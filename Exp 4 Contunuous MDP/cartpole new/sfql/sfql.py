import gym
import numpy as np
import dill
from tqdm import tqdm

np.random.seed(2)
env = gym.make("CartPole-v0")
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

from collections import defaultdict
import pickle

# Define a function to return [0]*4 as default value
def default_value():
    return [0] * 2

with open(r'LL_b1.pkl', 'rb') as handle:
    Q1 = dill.load(handle)
Q1 = defaultdict(default_value, Q1)

with open(r'LL_b2.pkl', 'rb') as handle:
    Q2 = dill.load(handle)
Q2 = defaultdict(default_value, Q2)


def set_buckets_and_actions():
    number_of_buckets =  (10,10,10,10)#buckets in each dimension
    number_of_actions = env.action_space.n
    
    #Creating a 2-tuple with the original bounds of each dimension
    # Convert tuples to lists
    state_value_bounds = [list(bounds) for bounds in zip(env.observation_space.low, env.observation_space.high)]
    state_value_bounds[0] = [-3,3]      
    state_value_bounds[1] = [-3.5,3.5]    
    state_value_bounds[2] = [-0.5,0.5]        
    state_value_bounds[3] = [-3.5,3.5]    

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



def q_learning(Q_sfql):
    Q = defaultdict(lambda: [0] * env.action_space.n)
    alpha = 0.1 #/ tile.numTilings
    for episodeNum in tqdm(range(numEpisodes)):
        G = 0
        step = 0
        state = env.reset()[0]
        step = 0
        while step<475:
            step+=1
            F = bucketize(state)
            action = np.argmax(Q_sfql[tuple(F)])  # Exploit
            next_state, reward, done, info, _ = env.step(action)
            G += reward
            if done:
                Q[tuple(F)][action] += alpha * (reward - Q[tuple(F)][action])
                break
            else:
                next_F = bucketize(next_state)
                Q[tuple(F)][action] += alpha * (reward + gamma * Q[tuple(next_F)][int(np.argmax(Q_sfql[tuple(next_F)]))] - Q[tuple(F)][action])
                state = next_state
            step += 1       
    return Q


gamma = 0.99
numEpisodes = 75000
Q1_new = q_learning(Q1)
with open('Q1_sfql.pickle', 'wb') as handle:
    dill.dump(Q1_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
Q2_new = q_learning(Q2)
with open('Q2_sfql.pickle', 'wb') as handle:
    dill.dump(Q2_new, handle, protocol=pickle.HIGHEST_PROTOCOL)

s1 = list(Q1_new.keys())

s2 = list(Q2_new.keys())

s=list(set(s1+s2))

Q_SFQL={}
for i in s:
    for a in range(2):
        Q_SFQL[i,a] = max(Q1_new[i,a], Q2_new[i,a])
    
with open('Q_SFQL_combined.pickle', 'wb') as handle:
    pickle.dump(Q_SFQL, handle, protocol=pickle.HIGHEST_PROTOCOL)
