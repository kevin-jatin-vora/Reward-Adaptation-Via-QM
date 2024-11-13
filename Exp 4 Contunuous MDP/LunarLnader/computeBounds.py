
import dill
import gym
import numpy as np
from collections import defaultdict

# Load the first pickle file
with open('behaviors//LL_b1.pkl', 'rb') as file:
    Q1 = dill.load(file)

# Load the second pickle file
with open('behaviors//LL_b2.pkl', 'rb') as file:
    Q2 = dill.load(file)
# Load the first pickle file
with open('behaviors//LL_b1_mu.pkl', 'rb') as file:
    Q1_mu = dill.load(file)

# Load the second pickle file
with open('behaviors//LL_b2_mu.pkl', 'rb') as file:
    Q2_mu = dill.load(file)

Q1 = defaultdict(lambda: [0] * 4, Q1)
Q2 = defaultdict(lambda: [0] * 4, Q2)
Q1_mu = defaultdict(lambda: [0] * 4, Q1_mu)
Q2_mu = defaultdict(lambda: [0] * 4, Q2_mu)

s1 = list(Q1_mu.keys())

s2 = list(Q2_mu.keys())

s3 = list(Q1.keys())

s4 = list(Q2.keys())


s=list(set(s1+s2+s3+s4))

S=len(s)
A=4
gamma = 0.99
from matplotlib import pyplot as plt
env = gym.make("LunarLander-v2")
import math

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
        



def check_done(F):
    state = de_bucketize(F)
    env = gym.make("LunarLander-v2")
    temp=env.reset()[0]
    
    env.lander.position.x = float(state[0])
    
    env.lander.position.y = float(state[1])
    
    env.lander.linearVelocity.x = float(state[2])
    
    env.lander.linearVelocity.y = float(state[3])
    
    env.lander.angle = float(state[4])
    
    env.lander.angularVelocity = float(state[5])
    
    env.legs[0].ground_contact = float(state[6])
    
    env.legs[1].ground_contact = float(state[7])
    
    next_state, reward, done, info, _ = env.step(0)
    
    env.close()
    
    if(done):
        return True
    else:
        return False


def transition(state, action):
    env = gym.make("LunarLander-v2")
    temp=env.reset()[0]
    
    env.lander.position.x = float(state[0])
    
    env.lander.position.y = float(state[1])
    
    env.lander.linearVelocity.x = float(state[2])
    
    env.lander.linearVelocity.y = float(state[3])
    
    env.lander.angle = float(state[4])
    
    env.lander.angularVelocity = float(state[5])
    
    env.legs[0].ground_contact = float(state[6])
    
    env.legs[1].ground_contact = float(state[7])
    
    next_state, reward, done, info, _ = env.step(action)
    
    env.close()
    
    if(done):
        next_state = env.reset()[0]
    
    return next_state, reward


def compute_sdash(F, action):
    state = de_bucketize(F)
    ns=[]
    reward=[]
    #getting 2 neighbours
    next_state, R = transition(state, action)
    ns.append((next_state,action))
    reward.append(R)
    
    next_state, R = transition(state, action)
    ns.append((next_state,action))
    reward.append(R)

    return tuple(zip(ns,reward))

#Q[terminal_state[0]*S + terminal_state[1],:]=0
Qm = np.zeros((len(s),A))
for i in range(S):
    for a in range(A):
        Qm[i,a] = Q1_mu[s[i]][a] + Q2_mu[s[i]][a] #max(Q1_mu[s[i]][a] + Q2[s[i]][a], Q1[s[i]][a] + Q2_mu[s[i]][a])
Q = np.zeros((len(s),A))
for i in range(S):
    for a in range(A):
        Q[i,a] = Q1[s[i]][a] + Q2[s[i]][a]

computed_sdash={}
r={}
done = []
for i in range(S):
    if(check_done(s[i])):
        done.append(s[i])
    for a in range(A):
        stof=[]
        for sdash,rew in compute_sdash(s[i],a):
            f_sdash = tuple(bucketize(sdash[0]))
            stof.append(f_sdash)
            r[(s[i],a,f_sdash)] = rew
            # R1[(s[i],a,f_sdash)] = Q1[s[i]][a] - gamma*np.max(Q1[f_sdash])
            # R2[(s[i],a,f_sdash)] = Q2[s[i]][a] - gamma*np.max(Q2[f_sdash])   
        computed_sdash[s[i],a] = stof


print("Starting Q-bounds-Refinement")
for j in range(50000):
    if(j%10==0):
        print(j)
    if(j>0):
        Um = Qm_k.copy()
        Udashm = Qm.copy()
    Qm_k=Qm.copy()
    Q_k = Q.copy()
    for i in range(S):
        if(s[i] in done):
            Q[i,:] = 0
            Qm[i,:] = 0
            continue
        else:
            for a in range(A):
                tempm=[]
                temp=[]
                for f_sdash in computed_sdash[s[i],a]:
                    # r1 = 0.5 #R1[(s[i],a,f_sdash)]
                    # r2 = 0.5 #R2[(s[i],a,f_sdash)]
                    #check if feature exists
                    if(f_sdash in s):
                        # print("idx available")
                        val = Q_k[s.index(f_sdash)]
                        valm = Qm_k[s.index(f_sdash)]
                    else:
                        val = 0
                        valm=0
                    tempm.append(r[(s[i],a,f_sdash)] + gamma*np.max(valm))
                    temp.append(r[(s[i],a,f_sdash)] + gamma*np.max(val))
                Q[i,a] =min(Q_k[i,a],max(temp)) #max(temp) 
                Qm[i,a] =max(Qm_k[i,a],min(tempm)) #max(temp)   
                if(Q[i,a]<Qm[i,a]):
                    Q[i,a] = Qm[i,a]
                if(Qm[i,a]>Q[i,a]):
                    Qm[i,a] = Q[i,a]
    if(j>1):
        if(np.round(np.max(np.abs(Qm_k-Qm)),7)> np.round(gamma*(np.max(np.abs(Um-Udashm))),7)):
            print(np.max(np.abs(Qm_k-Qm)))
            print(gamma*(np.max(np.abs(Um-Udashm))))
            print(j)
            # input()
    error1 = np.sum(np.abs(Qm-Qm_k))
    error2 = np.sum(np.abs(Q-Q_k))
    if(error1<0.0001 and error2<0.0001):
        print(j)
        print("------------")
        break
Q_UB={}
for i in range(len(Q)):
    Q_UB[s[i]] = Q[i]
import pickle
with open('Q_UB_test.pickle', 'wb') as handle:
    pickle.dump(Q_UB, handle, protocol=pickle.HIGHEST_PROTOCOL)
Q_LB={}
for i in range(len(Qm)):
    Q_LB[s[i]] = Qm[i]
with open('Q_LB_test.pickle', 'wb') as handle:
    pickle.dump(Q_LB, handle, protocol=pickle.HIGHEST_PROTOCOL)
