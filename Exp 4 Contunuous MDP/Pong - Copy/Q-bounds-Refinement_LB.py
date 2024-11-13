import dill
import gym
import numpy as np
from collections import defaultdict
action_size=3
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

s=list(set(s1+s2))

S=len(s)
A=3
gamma = 0.9

# Given information
width = 64
height = 48
cpu_speed_ratio = 0.6
players_speed_ratio = 0.4
ball_speed_ratio = 0.75

# Calculate paddle dimensions and distances
paddle_width = round(width * 0.023)
paddle_height = round(height * 0.15)
paddle_dist_to_wall = round(width * 0.0625)

# Calculate speeds
players_speed = players_speed_ratio * height
cpu_speed = cpu_speed_ratio * height
ball_speed = ball_speed_ratio * height

# Calculate maximum velocities
MAX_PADDLE_VELOCITY = players_speed_ratio * height
MAX_BALL_VELOCITY_X = ball_speed_ratio * height
MAX_BALL_VELOCITY_Y = ball_speed_ratio * height

# Calculate screen dimensions
SCREEN_WIDTH = width
SCREEN_HEIGHT = height
def set_buckets_and_actions():
    number_of_buckets = (3, 4, 3, 8, 8, 4, 4)   # Buckets in each dimension
    number_of_actions = 3  # Assuming Pong has 2 actions (up and down)
    
    # Set state value bounds based on the calculated values
    state_value_bounds = [
        [0, SCREEN_HEIGHT - PADDLE_HEIGHT],        # Player Y Position
        [-MAX_PADDLE_VELOCITY, MAX_PADDLE_VELOCITY], # Player Velocity
        [0, SCREEN_HEIGHT - PADDLE_HEIGHT],        # CPU Y Position
        [0, SCREEN_WIDTH],                          # Ball X Position
        [0, SCREEN_HEIGHT],                         # Ball Y Position
        [-MAX_BALL_VELOCITY_X, MAX_BALL_VELOCITY_X], # Ball X Velocity
        [-MAX_BALL_VELOCITY_Y, MAX_BALL_VELOCITY_Y]  # Ball Y Velocity
    ]
    
    return number_of_buckets, number_of_actions, state_value_bounds


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

# Calculate values
SCREEN_WIDTH = width
SCREEN_HEIGHT = height
PADDLE_HEIGHT = paddle_height
MAX_PADDLE_VELOCITY = players_speed_ratio * height
MAX_BALL_VELOCITY_X = ball_speed_ratio * height
MAX_BALL_VELOCITY_Y = ball_speed_ratio * height

# Set buckets and actions
number_of_buckets, number_of_actions, state_value_bounds = set_buckets_and_actions()


from ple import PLE
from ple.games.pong import Pong
import pygame
game = Pong(MAX_SCORE=5)
env = PLE(game, fps=30, display_screen=False)
def transition(state, action):
    env.reset_game()
    env.init()
    game.agentPlayer.pos.y = state[0]
    game.agentPlayer.vel.y = state[1]
    game.cpuPlayer.pos.y = state[2]
    game.ball.pos.x = state[3]
    game.ball.pos.y = state[4]
    game.ball.vel.x = state[5]
    game.ball.vel.y = state[6]
    if action == 0:  # Mapped action for moving paddle up
        r = env.act(119)
    elif action == 1:  # Mapped action for moving paddle down
        r = env.act(115)
    else:  # No action
        r = env.act(None)
    next_state = np.array(list(env.getGameState().values()))    
    return next_state


def compute_sdash(F, action):
    state = de_bucketize(F)
    ns=[]
    #for action in range(4):
    ns.append(transition(state, action))
    #for action in range(4):
    ns.append(transition(state, action))
    return ns

#Q[terminal_state[0]*S + terminal_state[1],:]=0
Q = np.zeros((len(s),action_size))
for i in range(S):
    for a in range(A):
        Q[i,a] = max(Q1_mu[s[i]][a] + Q2[s[i]][a], Q1[s[i]][a] + Q2_mu[s[i]][a])

computed_sdash={}
R1={}
R2={}
# done = []
from tqdm import tqdm
# for i in range(S):
for i in tqdm(range(S), desc="Outer Loop Progress"):
    # if(check_done(s[i])):
    #     done.append(s[i])
    for a in range(A):
        stof=[]
        for sdash in compute_sdash(s[i],a):
            f_sdash = tuple(bucketize(sdash))
            stof.append(f_sdash)
            R1[(s[i],a,f_sdash)] = Q1[s[i]][a] - gamma*np.max(Q1[f_sdash])
            R2[(s[i],a,f_sdash)] = Q2[s[i]][a] - gamma*np.max(Q2[f_sdash])   
        computed_sdash[s[i],a] = stof
pygame.quit()

print("Starting Q-bounds-Refinement [LB]: ")
for j in range(50000):
    if(j%10==0):
        print(j)
    if(j>0):
        U = Q_k.copy()
        Udash = Q.copy()
    Q_k=Q.copy()
    for i in range(S):
        # if(s[i] in done):
        #     Q[i,:] = 0
        #     continue
        # else:
        for a in range(A):
            temp=[]
            for f_sdash in computed_sdash[s[i],a]:
                r1 = R1[(s[i],a,f_sdash)]
                r2 = R2[(s[i],a,f_sdash)]
                # #check if feature exists
                # if(f_sdash in s):
                #     # print("idx available")
                #     val = Q_k[s.index(f_sdash)]
                # else:
                #     val = 0
                temp.append(r1 + r2 + gamma*np.max(Q_k[s.index(f_sdash)]))
            Q[i,a] =max(Q_k[i,a],min(temp)) #max(temp)             
    if(j>1):
        if(np.round(np.max(np.abs(Q_k-Q)),7)> np.round(gamma*(np.max(np.abs(U-Udash))),7)):
            print(np.max(np.abs(Q_k-Q)))
            print(gamma*(np.max(np.abs(U-Udash))))
            print(j)
            # input()
    error = np.sum(np.abs(Q-Q_k))
    if(error<0.0001):
        print(j)
        print("------------")
        break

Q_LB={}
for i in range(len(Q)):
    Q_LB[s[i]] = Q[i]
import pickle
with open('Q_LB.pickle', 'wb') as handle:
    pickle.dump(Q_LB, handle, protocol=pickle.HIGHEST_PROTOCOL)
