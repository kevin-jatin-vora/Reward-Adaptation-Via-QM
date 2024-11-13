import gym
import numpy as np
import dill
from tqdm import tqdm
np.random.seed(2)
from ple import PLE
from ple.games.pong import Pong
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
game = Pong(MAX_SCORE=5)
env = PLE(game, fps=30, display_screen=True)

from collections import defaultdict
import pickle

# Define a function to return [0]*4 as default value
def default_value():
    return [0] * 3

with open(r'LL_b1.pkl', 'rb') as handle:
    Q1 = dill.load(handle)
Q1 = defaultdict(default_value, Q1)

with open(r'LL_b2.pkl', 'rb') as handle:
    Q2 = dill.load(handle)
Q2 = defaultdict(default_value, Q2)

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
    state = state.reshape(-1,1)
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

print('State shape: ',len(env.getGameState()))
print('Number of actions: ', len(env.getActionSet()))


def q_learning(Q_sfql):
    Q = defaultdict(lambda: [0] * 3)
    alpha = 0.1 #/ tile.numTilings
    for episodeNum in tqdm(range(numEpisodes)):
        G = 0
        step = 0
        env.reset_game()
        env.init()
        state = np.array(list(env.getGameState().values())).reshape(1, 7)
        step = 0
        while step<500:
            step+=1
            F = bucketize(state)
            action = np.argmax(Q_sfql[tuple(F)])  # Exploit
            if action == 0:  # Mapped action for moving paddle up
                reward = env.act(119)
            elif action == 1:  # Mapped action for moving paddle down
                reward = env.act(115)
            else:  # No action
                reward = env.act(None)
            # vel = game.agentPlayer.vel.y
            # r_vel = -abs(vel) if abs(vel)<=0.2 else -25
            # r_vel = r_vel/3
            # reward*=2
            # reward+=r_vel
            next_state = np.array(list(env.getGameState().values()))
            done = env.game_over()
            G += reward
            if done:
                Q[tuple(F)][action] += alpha * (reward - Q[tuple(F)][action])
                break
            else:
                next_F = bucketize(next_state)
                Q[tuple(F)][action] += alpha * (reward + gamma * Q[tuple(next_F)][np.argmax(Q_sfql[tuple(next_F)])] - Q[tuple(F)][action])
                state = next_state
            step += 1       
    return Q


gamma = 0.99
numEpisodes = 2500
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
