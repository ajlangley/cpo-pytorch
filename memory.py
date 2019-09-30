from itertools import chain
import torch


class Trajectory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.done = False

    def __len__(self):
        return len(self.observations)


class Memory:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def sample(self):
        observations = torch.cat([torch.stack(trajectory.observations) for trajectory in self.trajectories])
        actions = torch.cat([torch.stack(trajectory.actions) for trajectory in self.trajectories])
        rewards = torch.cat([torch.tensor(trajectory.rewards) for trajectory in self.trajectories])
        costs = torch.cat([torch.tensor(trajectory.costs) for trajectory in self.trajectories])

        return observations, actions, rewards, costs

    def __getitem__(self, i):
        return self.trajectories[i]
