import copy
import itertools
import json

from src.envs.abs_env import AbstractEnv
import numpy as np
from farm_games_local.farmgym_games.game_catalogue.farm0.farm import env as env_maker
import gym


class Farm0(AbstractEnv):

    def __init__(self):
        self.ACTIONS = {
            "water_small": 0,
            "water_big": 1,
            "harvest": 3
        }

        self.env = env_maker()

        self.state_dim = 12
        self.observation_space = gym.spaces.Box(low=np.zeros((self.state_dim, )), high=np.array([1000] * self.state_dim), shape=(self.state_dim, ))
        self.action_space = gym.spaces.Discrete(3)

        self.max_penalty = -1

        self.max_feature = 365

    def step(self, action):
        action = self.decode_action(action)
        obs, rew, done, _, _ = self.env.step(action)

        flat_obs = self.flatten_obs(obs)

        self.state = np.array(flat_obs)

        return self.state, rew, done, {"composite_obs": obs}

    def reset(self):
        obs = self.env.reset()
        flat_obs = self.flatten_obs(obs)

        self.state = np.array(flat_obs)

        return self.state

    def render(self):
        print(self.state)

    def flatten_obs(self, obs):
        # take only the first element of the tuple that contains the obs -- only after reset
        obs = copy.copy(obs)
        if isinstance(obs, tuple):
            obs = obs[0]

        flat_obs = []
        # obs is a list of dictionaries
        for d in obs:
            while isinstance(d, dict):
                vals = list(d.values()) # there is only one key-value pair in all dicts
                d = vals[0]

            if isinstance(vals, list):
                for e in vals:
                    flat_obs.append(e)
            else:
                flat_obs.append(vals)

        return [e if not isinstance(e, list) else e[0] for e in flat_obs] # flatten list

    def decode_action(self, action):
        return [action]

    def get_actions(self, x):
        """ Returns a list of actions available in state x"""
        return np.arange(0, self.action_space.n)

    def set_state(self, x):
        #TODO: do this for all params not just ones passed by obs
        """ Changes the environment"s current state to x """
        farmer = self.env.farmers['BasicFarmer-0']
        field = farmer.fields['Field-0']

        sub_categories = ['air_temperature']

        x = self.unflatten_obs(x)

        for e_name, e in field.entities.items():
            vars = e.variables
            for v_name, v in vars.items():
                if isinstance(v, dict):
                    for sub_v_name, sub_v_val in v.items():
                        if sub_v_name in list(x.keys()) and v_name in sub_categories:
                            self.env.farmers['BasicFarmer-0'].fields['Field-0'].entities[e_name].variables[v_name][sub_v_name].value = x[sub_v_name]
                else:
                    if v_name in list(x.keys()):
                        if v_name == 'stage':
                            x[v_name] = self.env.farmers['BasicFarmer-0'].fields['Field-0'].entities[e_name].stages[int(x[v_name])]
                        try:
                            self.env.farmers['BasicFarmer-0'].fields['Field-0'].entities[e_name].variables[
                                v_name].value = x[v_name]
                        except AttributeError:
                            self.env.farmers['BasicFarmer-0'].fields['Field-0'].entities[e_name].variables[
                                v_name][0][0].value = x[v_name]


        self.state = copy.copy(x)

    def check_done(self, x):
        """ Returns a boolean indicating if x is a terminal state in the environment"""
        return False

    def equal_states(self, x1, x2):
        """ Returns a boolean indicating if x1 and x2 are the same state"""
        return sum(x1 != x2) == 0

    def writable_state(self, x):
        """ Returns a string with all state information to be used for writing results"""
        return [list(x)]

    def unflatten_obs(self, x):

        x = {'day#int365': x[0],
             'max#°C': x[1],
             'mean#°C': x[2],
             'min#°C': x[3],
             'sun_exposure#int5': x[4],
             'rain_amount': x[5],
             'consecutive_dry#day': x[6],
             'stage': x[7],
             'population#nb': x[8],
             'size#cm': x[9],
             'fruits_per_plant#nb': x[10],
             'fruit_weight#g': x[11]}

        return x



