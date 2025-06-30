import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import os
from joblib import load
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import MiniBatchKMeans

# Q-network remains unchanged
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ClusterBoundsManager adapted for MountainCar
class ClusterBoundsManager:
    def __init__(self, model_paths, device='cpu'):
        self.device = device
        self.kmeans = []
        self.bounds = []

        for path in model_paths:
            behavior_models = []
            behavior_bounds = []

            for action in [0, 1, 2]:
                model_data = load(os.path.join(path, f'action_{action}.pkl'))
                kmeans = model_data['kmeans']
                stats = model_data['stats']

                stats_sorted = sorted(stats, key=lambda x: x['cluster_label'])

                mins = torch.stack([torch.from_numpy(s['min']).float() for s in stats_sorted]).to(self.device)
                maxs = torch.stack([torch.from_numpy(s['max']).float() for s in stats_sorted]).to(self.device)

                behavior_models.append(kmeans)
                behavior_bounds.append({'min': mins, 'max': maxs})

            self.kmeans.append(behavior_models)
            self.bounds.append(behavior_bounds)

    def get_bounds(self, states, actions, behavior_idx=0):
        states_np = states.cpu().numpy() if isinstance(states, torch.Tensor) else states
        actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions

        n, state_dim = states.shape
        min_results = torch.zeros((n, state_dim), device=self.device)
        max_results = torch.zeros((n, state_dim), device=self.device)

        for action in [0, 1, 2]:
            mask = (actions_np == action)
            if mask.any():
                clusters = self.kmeans[behavior_idx][action].predict(states_np[mask])
                min_results[mask] = self.bounds[behavior_idx][action]['min'][clusters]
                max_results[mask] = self.bounds[behavior_idx][action]['max'][clusters]

        return min_results, max_results

    def get_combined_bounds(self, states, actions):
        all_mins = []
        all_maxs = []

        for i in range(len(self.kmeans)):
            min_b, max_b = self.get_bounds(states, actions, i)
            all_mins.append(min_b)
            all_maxs.append(max_b)

        combined_min = torch.stack(all_mins).min(dim=0)[0]
        combined_max = torch.stack(all_maxs).max(dim=0)[0]

        return combined_min, combined_max

class BoundsLearner:
    def __init__(self, state_size=2, action_size=3, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_low = torch.tensor([-1.2, -0.07], device=self.device)
        self.state_high = torch.tensor([0.51, 0.07], device=self.device)
        self.num_neighbors = 5

        model_paths = [
            'regression idea for neighbours//clustering_models',
            'regression idea for neighbours//clustering_models_2'
        ]
        self.cluster_manager = ClusterBoundsManager(model_paths, device=self.device)
        self._initialize_networks()

    def _initialize_networks(self):
        self.qnetwork_ub = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.qnetwork_lb = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.qnetwork_ub_target = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.qnetwork_lb_target = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)

        self.q1_ub = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.q2_ub = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.q1_lb = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
        self.q2_lb = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)

        for net, path in zip(
            [self.q1_ub, self.q2_ub, self.q1_lb, self.q2_lb],
            ['checkpoint_b1.pth', 'checkpoint_b2.pth', 'checkpoint_b1_mu.pth', 'checkpoint_b2_mu.pth']
        ):
            if os.path.exists(path):
                net.load_state_dict(torch.load(path, map_location=self.device))
                for param in net.parameters():
                    param.requires_grad = False

        self.optimizer_ub = optim.Adam(self.qnetwork_ub.parameters(), lr=0.0001)
        self.optimizer_lb = optim.Adam(self.qnetwork_lb.parameters(), lr=0.0001)
        self.soft_update(1.0)

    def initialize_networks(self, num_samples=3000000*2, batch_size=2048):
        print("Initializing networks...")
        ub_losses, lb_losses = [], []
        states = torch.empty((batch_size, self.state_size), device=self.device)

        for _ in range(num_samples // batch_size):
            states.uniform_(-1, 1)
            states = states * (self.state_high - self.state_low) / 2 + (self.state_high + self.state_low) / 2

            with torch.no_grad():
                ub_targets = self.q1_ub(states) + self.q2_ub(states)
                lb_targets = self.q1_lb(states) + self.q2_lb(states)
                ub_targets = torch.max(lb_targets, ub_targets)
                lb_targets = torch.min(lb_targets, ub_targets)

            self.optimizer_ub.zero_grad()
            ub_loss = F.mse_loss(self.qnetwork_ub(states), ub_targets)
            ub_loss.backward()
            self.optimizer_ub.step()
            ub_losses.append(ub_loss.item())

            self.optimizer_lb.zero_grad()
            lb_loss = F.mse_loss(self.qnetwork_lb(states), lb_targets)
            lb_loss.backward()
            self.optimizer_lb.step()
            lb_losses.append(lb_loss.item())

        print(f"UB init loss: {np.mean(ub_losses):.4f} | LB init loss: {np.mean(lb_losses):.4f}")
        torch.save(self.qnetwork_ub.state_dict(), 'UB_init.pth')
        torch.save(self.qnetwork_lb.state_dict(), 'LB_init.pth')

    def get_neighbors(self, states, actions):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self.cluster_manager.get_bounds, states, actions, 0)
            future2 = executor.submit(self.cluster_manager.get_bounds, states, actions, 1)
            min1, max1 = future1.result()
            min2, max2 = future2.result()

        min_deltas, max_deltas = self.cluster_manager.get_combined_bounds(states, actions)
        rand_vals = torch.rand((len(states), self.num_neighbors, self.state_size), device=self.device)
        deltas = min_deltas.unsqueeze(1) + rand_vals * (max_deltas - min_deltas).unsqueeze(1)
        next_states = states.unsqueeze(1) + deltas
        return torch.clamp(next_states, self.state_low, self.state_high)

    def compute_rewards(self, next_states):
        position = next_states[..., 0]
        done = position >= 0.5
        reward = torch.full_like(position, -1.0)
        return reward, done

    def joint_update(self, batch_size=1024, gamma=0.99):
        states = torch.rand((batch_size, self.state_size), device=self.device)
        states = states * (self.state_high - self.state_low) + self.state_low
        actions = torch.randint(0, self.action_size, (batch_size,), device=self.device)
        next_states = self.get_neighbors(states, actions)
        rewards, dones = self.compute_rewards(next_states)

        current_q_ub = self.qnetwork_ub(states).gather(1, actions.unsqueeze(1))
        current_q_lb = self.qnetwork_lb(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            flat_next = next_states.view(-1, self.state_size)
            next_q_ub = self.qnetwork_ub_target(flat_next).view(batch_size, self.num_neighbors, -1)
            next_q_lb = self.qnetwork_lb_target(flat_next).view(batch_size, self.num_neighbors, -1)

            max_next_ub = next_q_ub.max(2)[0]
            max_next_lb = next_q_lb.max(2)[0]
            max_next_ub[dones] = 0
            max_next_lb[dones] = 0

            targets_ub = (rewards + gamma * max_next_ub).max(1)[0].unsqueeze(1)
            targets_lb = (rewards + gamma * max_next_lb).min(1)[0].unsqueeze(1)

            targets_ub = torch.maximum(targets_ub, targets_lb)
            targets_lb = torch.minimum(targets_lb, targets_ub)
            targets_ub = torch.minimum(targets_ub, current_q_ub)
            targets_lb = torch.maximum(targets_lb, current_q_lb)

        self.optimizer_ub.zero_grad()
        ub_loss = F.mse_loss(current_q_ub, targets_ub)
        penalty_ub = F.relu(current_q_lb.detach() - current_q_ub).pow(2).mean()
        (ub_loss + penalty_ub).backward()
        self.optimizer_ub.step()

        self.optimizer_lb.zero_grad()
        lb_loss = F.mse_loss(current_q_lb, targets_lb)
        penalty_lb = F.relu(current_q_lb - current_q_ub.detach()).pow(2).mean()
        (lb_loss + penalty_lb).backward()
        self.optimizer_lb.step()

        return ub_loss.item(), lb_loss.item()

    def soft_update(self, tau=0.001):
        with torch.no_grad():
            for target, source in zip(
                [self.qnetwork_ub_target, self.qnetwork_lb_target],
                [self.qnetwork_ub, self.qnetwork_lb]
            ):
                for t_param, s_param in zip(target.parameters(), source.parameters()):
                    t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

    def train(self, num_episodes=25000, update_every=4):
        init_paths = ['UB_init.pth', 'LB_init.pth']
        s1=time.time()
        if all(os.path.exists(p) for p in init_paths):
            self.qnetwork_ub.load_state_dict(torch.load(init_paths[0], map_location=self.device))
            self.qnetwork_lb.load_state_dict(torch.load(init_paths[1], map_location=self.device))
        else:
            self.initialize_networks()
        e1=time.time()
        file_path = "init_time.txt"        
        with open(file_path, 'w') as file:
            file.write(str(e1-s1))
        print("Starting training...")
        start_time = time.time()
        s2=time.time()
        for episode in range(num_episodes):
            ub_loss, lb_loss = self.joint_update()

            if episode % update_every == 0:
                self.soft_update()

            if episode % 100 == 0:
                elapsed = time.time() - start_time
                eps_remaining = num_episodes - episode - 1
                time_remaining = elapsed / (episode + 1) * eps_remaining if episode > 0 else 0
                print(f"Episode {episode}: UB Loss {ub_loss:.4f}, LB Loss {lb_loss:.4f} | Elapsed: {elapsed:.2f}s | Remaining: {time_remaining:.2f}s")

        torch.save(self.qnetwork_ub.state_dict(), 'UB_clustering.pth')
        torch.save(self.qnetwork_lb.state_dict(), 'LB_clustering.pth')
        e2=time.time()
        file_path = "iter_time.txt"        
        with open(file_path, 'w') as file:
            file.write(str(e2-s2))

if __name__ == "__main__":
    start_time = time.time()
    learner = BoundsLearner()
    learner.train(num_episodes=8000)
    print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")
    file_path = "total_time.txt"        
    with open(file_path, 'w') as file:
        file.write(str(time.time()-start_time))
