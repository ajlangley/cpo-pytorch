from collections import namedtuple
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from mujoco_py.generated.const import GEOM_SPHERE
import numpy as np
from numpy import pi as pi
import tempfile
import xml.etree.ElementTree as ET

from autoassign import autoassign
from envs.mujoco_utils import euclidian_dist, in_range, std_radians

APPLE = 0
BOMB = 1

APPLE_RGBA = np.asarray([0.7, 0.13, 0.17, 1.0])
BOMB_RGBA = np.asarray([0.22, 0.6, 0.18, 1.0])

Object = namedtuple('Object', 'type x y')


class GatherEnv:
    @autoassign(exclude=('model_path'))
    def __init__(self, model_path, n_apples=8, n_bombs=8, apple_reward=10,
                 bomb_cost=1, activity_range=6.0, catch_range=10, n_bins=10,
                 robot_object_spacing=2.0, sensor_range=6.0, sensor_span=pi):
        self.viewer = None

        self.apples = []
        self.bombs = []

        self.model = self.build_model(model_path)
        self.sim = MjSim(self.model)

        self._max_episode_steps = None
        self._step_num = 0

    def build_model(self, agent_xml_path):
        sim_size = self.activity_range + 1

        model_tree = ET.parse(agent_xml_path)
        worldbody = model_tree.find('.//worldbody')

        floor_attrs = dict(name='floor', type='plane', material='MatPlane',
                           pos='0 0 0', size=f'{sim_size} {sim_size} {sim_size}',
                           conaffinity='1', rgba='0.8 0.9 0.8 1', condim='3')
        wall_attrs = dict(type='box', conaffinity='1', rgba='0.8 0.9 0.8 1',
                          condim='3')

        wall_poses = [f'0 {-sim_size} 0', f'0 {sim_size} 0', \
                      f'{-sim_size} 0 0', f'{sim_size} 0 0']
        wall_sizes = [f'{sim_size + 0.1} 0.1 1', f'{sim_size + 0.1} 0.1 1', \
                      f'0.1 {sim_size + 0.1} 1', f'0.1 {sim_size + 0.1} 1']

        # Add a floor to the model
        ET.SubElement(worldbody, 'geom', **floor_attrs)

        # Add walls to the model
        for i, pos, size in zip(range(4), wall_poses, wall_sizes):
            ET.SubElement(worldbody,
                          'geom',
                          dict(**wall_attrs,
                               name=f'wall{i}',
                               pos=pos,
                               size=size))

        model_xml = ET.tostring(model_tree.getroot(), encoding='unicode', method='xml')
        model = load_model_from_xml(model_xml)

        return model

    def reset(self):
        self.sim.reset()

        self.apples = []
        self.bombs = []
        obj_coords = []

        self._step_num = 0

        agent_x_pos, agent_y_pos = self.sim.data.qpos[0], self.sim.data.qpos[1]

        rand_coord = lambda: np.random.randint(-self.activity_range, self.activity_range)

        while (len(self.apples) < self.n_apples):
            x, y = rand_coord(), rand_coord()

            # Change this later to make (0, 0) the position of the agent
            if in_range(x, y, agent_x_pos, agent_y_pos, self.robot_object_spacing):
                continue

            if (x, y) in obj_coords:
                continue

            self.apples.append(Object(APPLE, x, y))
            obj_coords.append((x, y))

        while (len(self.bombs) < self.n_bombs):
            x, y = rand_coord(), rand_coord()

            if in_range(x, y, 0, 0, self.robot_object_spacing):
                continue

            if (x, y) in obj_coords:
                continue

            self.bombs.append(Object(BOMB, x, y))
            obj_coords.append((x, y))

        return self._get_obs()

    def step(self, action):
        raise NotImplementedError

    def _do_simulation(self, action):
        raise NotImplementedError

    def _update_objects(self):
        n_apples = 0
        n_bombs = 0
        agent_x_pos, agent_y_pos = self.sim.data.qpos[0], self.sim.data.qpos[1]

        for apple in self.apples:
            if in_range(apple.x, apple.y, agent_x_pos, agent_y_pos, self.catch_range):
                n_apples += 1
                self.apples.remove(apple)

        for bomb in self.bombs:
            if in_range(bomb.x, bomb.y, agent_x_pos, agent_y_pos, self.catch_range):
                n_bombs += 1
                self.bombs.remove(bomb)

        return n_apples, n_bombs

    def _get_self_obs(self):
        raise NotImplementedError

    def _get_sensor_obs(self):
        apple_readings = np.zeros(self.n_bins)
        bomb_readings = np.zeros(self.n_bins)

        idx = self.model.body_names.index('torso')
        com = self.sim.data.subtree_com[idx].flat
        agent_x, agent_y = com[:2]

        sort_key = lambda obj: -euclidian_dist(obj.x, obj.y, agent_x, agent_y)
        sorted_objs = sorted(self.apples + self.bombs, key=sort_key)

        bin_res = self.sensor_span / self.n_bins
        half_span = self.sensor_span / 2
        orientation = std_radians(self._get_orientation())

        for obj_type, obj_x, obj_y in sorted_objs:
            dist = euclidian_dist(obj_x, obj_y, agent_x, agent_y)

            if dist > self.sensor_range:
                continue

            # Get the components of the vector from the agent to the object
            x_delta = obj_x - agent_x
            y_delta = obj_y - agent_y

            # Get the angle of the vector relative to the agent's sensor range
            angle = np.arctan2(y_delta, x_delta)
            angle = std_radians(angle - orientation + half_span)

            if angle > self.sensor_span:
                continue

            bin_number = int(angle / bin_res)

            if bin_number == int(angle / bin_res):
                bin_number -= 1

            intensity = 1.0 - dist / self.sensor_range

            if obj_type == APPLE:
                apple_readings[bin_number] = intensity
            else:
                bomb_readings[bin_number] = intensity

        sensor_obs = np.concatenate([apple_readings, bomb_readings])

        return sensor_obs

        #     dist = euclidian_dist(obj_x, obj_y, agent_x, agent_y)
        #
        #     if dist > self.sensor_range:
        #         continue
        #
        #     # Get the components of the vector from the agent to the object
        #     x_comp = obj_x - agent_x
        #     y_comp = obj_y - agent_y
        #
        #     # Get the angle of the vector relative to the agent's sensor range
        #     angle = np.arctan2(y_comp, x_comp)
        #     angle = standardize_radians(angle - orientation)
        #
        #     # Skip if object is outside sensor span
        #     if angle > half_span and angle < 2 * pi - half_span:
        #         continue
        #
        #     # Standardize angle to range [0, pi] to more easily find bin number
        #     angle = standardize_radians(angle + half_span)
        #     bin_number = int(angle / bin_res)
        #
        #     # The object is exactly on the upper bound of the sensor span
        #     if bin_number == self.n_bins:
        #         bin_number -= 1
        #
        #     intensity = 1.0 - dist / self.sensor_range
        #
        #     if obj_type == APPLE:
        #         apple_readings[bin_number] = intensity
        #     else:
        #         bomb_readings[bin_number] = intensity
        #
        # sensor_obs = np.concatenate([apple_readings, bomb_readings])
        #
        # return sensor_obs

    def _get_obs(self):
        return np.concatenate([self._get_self_obs(), self._get_sensor_obs()])

    def _get_orientation(self):
        raise NotImplementedError

    def _is_done(self):
        raise NotImplementedError

    def _unhealthy_cost(self):
        raise NotImplementedError

    def render(self):
        if not self.viewer:
            self.viewer = MjViewer(self.sim)

        objects = self.apples + self.bombs

        for object in objects:
            x, y = object.x, object.y
            rgba = APPLE_RGBA if object.type is APPLE else BOMB_RGBA
            self.viewer.add_marker(type=GEOM_SPHERE,
                                   pos=np.asarray([x, y, 0.5]),
                                   rgba=rgba,
                                   size=np.asarray([0.5] * 3))

        self.viewer.render()
