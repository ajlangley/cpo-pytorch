import numpy as np

from envs.gather_env import GatherEnv


class PointGatherEnv(GatherEnv):
    def __init__(self, n_apples=8, n_bombs=8, apple_reward=10, bomb_cost=1,
                 activity_range=6.0, catch_range=1.0, n_bins=10,
                 robot_object_spacing=2.0, sensor_range=6.0, sensor_span=np.pi):
        GatherEnv.__init__(self, 'envs/assets/point.xml', n_apples, n_bombs,
                           apple_reward, bomb_cost, activity_range, catch_range,
                           n_bins, robot_object_spacing, sensor_range,
                           sensor_span)

    def step(self, action):
        self._do_simulation(action)
        n_apples, n_bombs = self._update_objects()

        cost = n_bombs * self.bomb_cost
        reward = n_apples * self.apple_reward - cost
        info = dict(reward=reward, constraint_cost=cost)

        self._step_num += 1

        return self._get_obs(), reward, self._is_done(), info

    def _do_simulation(self, action):
        qpos = np.copy(self.sim.data.qpos)
        qpos[2] += action[1]
        orientation = qpos[2]

        dx = np.cos(orientation) * action[0]
        dy = np.sin(orientation) * action[0]

        qpos[0] = np.clip(qpos[0] + dx, -self.activity_range, self.activity_range)
        qpos[1] = np.clip(qpos[1] + dy, -self.activity_range, self.activity_range)
        self.sim.data.qpos[:] = qpos
        self.sim.forward()

    def _get_self_obs(self):
        idx = self.model.body_names.index('torso')
        pos = self.sim.data.qpos.flat
        vel = self.sim.data.qvel.flat
        com = self.sim.data.subtree_com[idx].flat
        obs = np.concatenate([pos, vel, com])

        return obs

    def _get_orientation(self):
        return self.sim.data.qpos[2]

    def _is_done(self):
        return self._max_episode_steps and self._step_num >= self._max_episode_steps
