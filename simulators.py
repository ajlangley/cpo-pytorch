from collections import defaultdict
import gym
from gym.spaces import Box, Discrete
import numpy as np
import torch

from memory import Memory
from torch_utils import get_device


class Simulator:
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, state_filter=None,
                **env_args):
        self.env = np.asarray([gym.make(env_name, **env_args) for i in range(n_trajectories)])

        for env in self.env:
            env._max_episode_steps = trajectory_len

        self.action_space = self.env[0].action_space
        self.policy = policy
        self.n_trajectories = n_trajectories
        self.trajectory_len = trajectory_len
        self.state_filter = state_filter
        self.device = get_device()


class SinglePathSimulator(Simulator):
    def __init__(self, env_name, policy, n_trajectories, trajectory_len, state_filter=None,
                **env_args):
        Simulator.__init__(self, env_name, policy, n_trajectories, trajectory_len, state_filter,
                           **env_args)

    def sample_trajectories(self):
        self.policy.eval()

        with torch.no_grad():
            memory = np.asarray([defaultdict(list) for i in range(self.n_trajectories)])
            done = [False] * self.n_trajectories
            for trajectory in memory:
                trajectory['done'] = False

            for env, trajectory in zip(self.env, memory):
                state = torch.tensor(env.reset()).float()

                if self.state_filter:
                    state = self.state_filter(state)

                trajectory['states'].append(state)

            while not np.all(done):
                continue_mask = [i for i, trajectory in enumerate(memory) if not trajectory['done']]
                trajs_to_update = [trajectory for trajectory in memory if not trajectory['done']]
                continuing_envs = [env for i, env in enumerate(self.env) if i in continue_mask]

                policy_input = torch.stack([torch.tensor(trajectory['states'][-1]).to(self.device)
                                            for trajectory in trajs_to_update])

                action_dists = self.policy(policy_input)
                actions = action_dists.sample()
                actions = actions.cpu()

                for env, action, trajectory in zip(continuing_envs, actions, trajs_to_update):
                    state, reward, done, info = env.step(action.numpy())

                    state = torch.tensor(state).float()
                    reward = torch.tensor(reward, dtype=torch.float)

                    if self.state_filter:
                        state = self.state_filter(state)

                    trajectory['actions'].append(action)
                    trajectory['rewards'].append(reward)
                    trajectory['done'] = done

                    if not done:
                        trajectory['states'].append(state)

                done = [trajectory['done'] for trajectory in memory]

        return memory
