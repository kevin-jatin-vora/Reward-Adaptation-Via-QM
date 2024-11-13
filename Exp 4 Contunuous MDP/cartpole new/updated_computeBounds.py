import dill
import gym
import numpy as np
from collections import defaultdict
action_size=2
# Load the first pickle file
with open('behaviors//LL_b1.pkl', 'rb') as file:
    Q1 = dill.load(file)

# Load the second pkl file
with open('behaviors//LL_b2.pkl', 'rb') as file:
    Q2 = dill.load(file)
# Load the first pkl file
with open('behaviors//LL_b1_mu.pkl', 'rb') as file:
    Q1_mu = dill.load(file)

# Load the second pkl file
with open('behaviors//LL_b2_mu.pkl', 'rb') as file:
    Q2_mu = dill.load(file)

s1 = list(Q1_mu.keys())

s2 = list(Q2_mu.keys())

s3 = list(Q1.keys())

s4 = list(Q2.keys())


s=list(set(s1+s2+s3+s4))

# def generate_tuples(ranges):
#   """Generates all possible tuples based on given ranges.

#   Args:
#     ranges: A list of ranges for each element in the tuple.

#   Returns:
#     A list of tuples representing all possible combinations.
#   """

#   if not ranges:
#     return [()]  # Base case: empty ranges yield an empty tuple

#   tuples = []
#   for value in range(ranges[0][0], ranges[0][1]):
#     subtuples = generate_tuples(ranges[1:])  # Recursively generate subtuples
#     for subtuple in subtuples:
#       tuples.append((value, *subtuple))  # Combine value with subtuples

#   return tuples

# # Define the ranges for each element
# ranges = [(0, 10), (0, 10), (0, 10), (0, 10)]

# # Generate and print the tuples
# s = generate_tuples(ranges)

S=len(s)
A=2
gamma = 0.9
env = gym.make("CartPole-v0")

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

def de_bucketize(bucket_indexes):
    state = []
    for i in range(len(bucket_indexes)):
        # Calculate the range of values for the current bucket
        bucket_min = state_value_bounds[i][0] + (bucket_indexes[i] / (number_of_buckets[i] - 1)) * (state_value_bounds[i][1] - state_value_bounds[i][0])
        bucket_max = state_value_bounds[i][0] + ((bucket_indexes[i] + 1) / (number_of_buckets[i] - 1)) * (state_value_bounds[i][1] - state_value_bounds[i][0])
        
        # Calculate the center of the bucket
        bucket_center = (bucket_min + bucket_max) / 2
        
        state.append(bucket_center)
    return state


def check_termination(next_state):
    # Define the range for cart x-position and pole angle
    cart_position_range = (-2.4, 2.4)
    pole_angle_range = (-0.2095, 0.2095)

    # Extract cart x-position and pole angle from the next state
    cart_position = next_state[0]
    pole_angle = next_state[2]

    # Check if the cart x-position or pole angle is outside the termination range
    cart_out_of_range = cart_position < cart_position_range[0] or cart_position > cart_position_range[1]
    pole_out_of_range = pole_angle < pole_angle_range[0] or pole_angle > pole_angle_range[1]

    # If either condition is met, the episode is terminated
    if cart_out_of_range or pole_out_of_range:
        return True  # Terminated
    else:
        return False  # Not terminated
    

def check_done(F):
    state = de_bucketize(F)
    done1 = check_termination(state)
    env.close()
    
    if(done1):
        return True
    else:
        return False


def transition(state, action):
    env = gym.make("CartPole-v0")
    temp=env.reset()[0]
    
    env.state = state
    next_state, reward, done, info, _ = env.step(action)
    done1 = check_termination(next_state)
    env.close()
    
    if(done1):
        next_state = env.reset()[0]
    
    return next_state


def compute_sdash(F, action):
    state = de_bucketize(F)
    ns=[]
    #for action in range(4):
    ns.append(transition(state, action))
    #for action in range(4):
    ns.append(transition(state, action))
    return ns

# Q[terminal_state[0]*S + terminal_state[1],:]=0
Qm = np.zeros((len(s),action_size))
for i in range(S):
    for a in range(A):
        Qm[i,a] = Q1_mu[s[i]][a] + Q2_mu[s[i]][a] #max(Q1_mu[s[i]][a] + Q2[s[i]][a], Q1[s[i]][a] + Q2_mu[s[i]][a]) -200
Q = np.zeros((len(s),action_size))
for i in range(S):
    for a in range(A):
        Q[i,a] = Q1[s[i]][a] + Q2[s[i]][a] +200

# Q = np.random.uniform(200,201,(S,A))
# Qm = np.random.uniform(-200,-201,(S,A))

computed_sdash={}
R1={}
R2={}
done = []
from tqdm import tqdm

# for i in range(S):
for i in tqdm(range(S), desc="Outer Loop Progress"):
    if(check_done(s[i])):
        done.append(s[i])
    for a in range(A):
        stof=[]
        for sdash in compute_sdash(s[i],a):
            f_sdash = tuple(bucketize(sdash))
            stof.append(f_sdash)
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
                    r1 = 0.5 #R1[(s[i],a,f_sdash)]
                    r2 = 0.5 #R2[(s[i],a,f_sdash)]
                    # #check if feature exists
                    # if(f_sdash in s):
                    #     # print("idx available")
                    #     val = Q_k[s.index(f_sdash)]
                    # else:
                    #     val = 0
                    tempm.append(r1 + r2 + gamma*np.max(Qm_k[s.index(f_sdash)]))
                    temp.append(r1 + r2 + gamma*np.max(Q_k[s.index(f_sdash)]))
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
    error1 = np.abs(np.sum(Qm-Qm_k))
    error2 = np.abs(np.sum(Q-Q_k))
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
