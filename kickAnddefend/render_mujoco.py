import collections
import ctypes
import math
import os.path as osp
import re

from PIL import Image, ImageDraw, ImageFont
import gym
import mujoco_py_131
import numpy as np
import pdb
from collections import Counter, OrderedDict, defaultdict
import os, json

VICTIM_INDEX = collections.defaultdict(lambda: 0)



VICTIM_OPPONENT_COLORS = {
    'Player_0': (55, 126, 184, 255),
    'Player_1': (228, 26, 28, 255),
    'Ties': (0, 0, 0, 255),
}


def body_color(is_victim, is_masked, agent_type, agent_path):
    key = 'Player_0' if is_victim else 'Player_1'
    return VICTIM_OPPONENT_COLORS[key]


GEOM_MAPPINGS = {
    '*': body_color,
}

def env_name_to_canonical(env_name):
    env_aliases = {
        'multicomp/SumoHumansAutoContact-v0': 'multicomp/SumoHumans-v0',
        'multicomp/SumoAntsAutoContact-v0': 'multicomp/SumoAnts-v0',
    }
    env_name = env_aliases.get(env_name, env_name)
    env_prefix, env_suffix = env_name.split('/')
    if env_prefix != 'multicomp':
        raise ValueError(f"Unsupported env '{env_name}'; must start with multicomp")
    return env_suffix

def set_geom_rgba(model, value):
    """Does what model.geom_rgba = ... should do, but doesn't because of a bug in mujoco-py."""
    val_ptr = np.array(value, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ctypes.memmove(model._wrapped.contents.geom_rgba, val_ptr,
                   model.ngeom * 4 * ctypes.sizeof(ctypes.c_float))


def set_geom_colors(model, patterns):
    names = [name.decode('utf-8') for name in model.geom_names]
    patterns = {re.compile(k): tuple(x / 255 for x in v) for k, v in patterns.items()}

    modified = np.array(model.geom_rgba)
    for row_idx, name in enumerate(names):
        for pattern, color in patterns.items():
            if pattern.match(name):
                modified[row_idx, :] = color

    set_geom_rgba(model, modified)

CAMERA_CONFIGS = {
    # For website videos
    'default': {
        # From behind
        'KickAndDefend-v0': {'azimuth': 0, 'distance': 10, 'elevation': -16.5},
        # From side, slightly behind (runner always goes forward, never back)
        'YouShallNotPassHumans-v0': {'azimuth': 140, 'distance': 9, 'elevation': -21,
                                     'lookat': [-1.5, 0.5, 0.0], 'trackbodyid': -1},
        # From side, close up
        'SumoHumans-v0': {'azimuth': 90, 'distance': 8, 'elevation': -23},
        'SumoAnts-v0': {'azimuth': 90, 'distance': 10, 'elevation': -25},
    },
    # More closely cropped. May miss action, but easier to see on projector.
    'close': {
        'KickAndDefend-v0': {'azimuth': 0, 'distance': 10, 'elevation': -15},
        'YouShallNotPassHumans-v0': {'azimuth': 150, 'distance': 9, 'elevation': -23,
                                     'lookat': [-2.0, 1, 0.0], 'trackbodyid': -1},
        'SumoHumans-v0': {'azimuth': 90, 'distance': 7.2, 'elevation': -22.5},
        'SumoAnts-v0': {'azimuth': 90, 'distance': 10, 'elevation': -25},
    },
    # Camera tracks victim. Very tightly cropped. May miss what opponent is doing.
    'track': {
        'KickAndDefend-v0': {'azimuth': 0, 'distance': 7, 'elevation': -25,
                             'trackbodyid': 'agent0/torso'},
        'YouShallNotPassHumans-v0': {'azimuth': 140, 'distance': 5, 'elevation': -30,
                                     'trackbodyid': 'agent1/torso'},
        'SumoHumans-v0': {'azimuth': 90, 'distance': 7, 'elevation': -30},
        'SumoAnts-v0': {'azimuth': 90, 'distance': 10, 'elevation': -25},
    },
}


class Render_mujoco(gym.Wrapper):
    metadata = {
        'video.frames_per_second': 60,  # MuJoCo env default FPS is 67, round down to be standard
    }

    def __init__(self, env, env_name, mask_agent_index, resolution, camera_config, draw=True):
        super(Render_mujoco, self).__init__(env)

        # Set agent colors
        self.env_name = env_name
        self.victim_index = VICTIM_INDEX[env_name]
        # self.victim_index = 0
        self.mask_agent_index = mask_agent_index
       
        self.agent_mapping = {
            0: (0 == self.victim_index, 0 == self.mask_agent_index,
                None, None),
            1: (1 == self.victim_index, 1 == self.mask_agent_index,
                None, None),
        }

        # Camera settings
        self.camera_config = CAMERA_CONFIGS[camera_config]

        # Internal state
        self.result = collections.defaultdict(int)
        self.changed = collections.defaultdict(int)
        self.last_won = None
        self.draw = draw

        env_scene = self.env.unwrapped.env_scene

        # Start the viewer ourself to control dimensions.
        # env_scene only sets this if None so will not be overwritten.
        width, height = resolution
        env_scene.viewer = mujoco_py_131.MjViewer(init_width=width, init_height=height)
        env_scene.viewer.start()
        env_scene.viewer.set_model(env_scene.model)
        env_scene.viewer_setup()
        self.camera_setup()

    def camera_setup(self):
        # Color mapping
        model = self.env.unwrapped.env_scene.model
        color_patterns = {f'agent{agent_key}/{geom_key}': geom_fn(*agent_val)
                          for geom_key, geom_fn in GEOM_MAPPINGS.items()
                          for agent_key, agent_val in self.agent_mapping.items()}
        set_geom_colors(model, color_patterns)

        # Camera setup
        canonical_env_name = env_name_to_canonical(self.env_name)
        camera_cfg = self.camera_config[canonical_env_name]

        if 'trackbodyid' in camera_cfg:
            trackbodyid = camera_cfg['trackbodyid']
            try:
                trackbodyid = int(trackbodyid)
            except ValueError:
                trackbodyid = str(trackbodyid).encode('utf-8')
                trackbodyid = model.body_names.index(trackbodyid)
            camera_cfg['trackbodyid'] = trackbodyid

        if 'lookat' in camera_cfg:
            DoubleArray3 = ctypes.c_double * 3
            lookat = [float(x) for x in camera_cfg['lookat']]
            assert len(lookat) == 3
            camera_cfg['lookat'] = DoubleArray3(*lookat)

        viewer = self.env.unwrapped.env_scene.viewer
        for k, v in camera_cfg.items():
            setattr(viewer.cam, k, v)

    def reset(self):
        ob = super(Render_mujoco, self).reset()
        if self.env.unwrapped.env_scene.viewer is not None:
            self.camera_setup()
        return ob

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

    def render(self, mode='human', close=False):
        res = self.env.render(mode)
        if mode == 'rgb_array':
           return res
