import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from gym.wrappers import RecordVideo

# Q-Network for MountainCar-v0
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

# Setup
id=2
state_size = 2
action_size = 3
seed = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load trained model
qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
qnetwork_local.load_state_dict(torch.load(f'checkpoint_b{id}.pth', map_location=device))
qnetwork_local.eval()

# Create environment and enable video recording
env = gym.make("MountainCar-v0", render_mode='rgb_array')
env = RecordVideo(env, f'video_b{id}', episode_trigger=lambda ep: 3 <= ep <= 6)

episodes = 10
for i_episode in range(1, episodes + 1):
    state = env.reset()[0]
    for t in range(200):  # 200 is the default max steps in MountainCar
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            action_values = qnetwork_local(st)
        action = np.argmax(action_values.cpu().data.numpy())

        next_state, reward, done, _, _ = env.step(action)
        state = next_state

        if done:
            break

env.close()
