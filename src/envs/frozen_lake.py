import random

import gym

from src.envs.abs_env import AbstractEnv
import numpy as np

FROZEN_SQUARES = [2, 12, 23, 15, 14]


class FrozenLake(AbstractEnv):
    def __init__(self):
        super(FrozenLake, self).__init__()

        self.world_dim = 5
        self.state_dim = 7

        self.lows = np.zeros((self.state_dim,))
        self.highs = np.ones((self.state_dim,))
        self.highs = np.array([self.world_dim] * self.state_dim)

        self.observation_space = gym.spaces.Box(low=self.lows, high=self.highs, shape=(self.state_dim, ))
        self.action_space = gym.spaces.Discrete(5)

        self.state = np.zeros((10, ))

        self.steps = 0
        self.max_steps = 200

        self.max_feature = 24

        self.FROZEN_SQUARES = FROZEN_SQUARES
        self.COSTLY_SQUARES = []
        self.GOAL_STATES = list(np.arange(0, self.world_dim**2))
        self.AGENT_START_STATES = list(np.arange(0, self.world_dim**2))

        self.ACTIONS = {'LEFT': 0, 'DOWN': 1, 'RIGHT': 2, 'UP': 3, 'EXIT': 4}
        self.REWARDS = {'step': -1, 'costly step': -5, 'goal': +10, 'wrong_goal': -3}

        self.max_penalty = min(list(self.REWARDS.values()))

    def step(self, action):
        agent = self.state['agent']
        goal = self.state['goal']
        frozen = self.state['frozen']
        costly = self.state['costly']

        if agent in frozen:
            move_prob = np.random.randint(0, 2)
            # if agent is on frozen tile there is probability the state stays the same
            if not move_prob:
                random_action = np.random.choice([0, 1, 2, 3])
                action = random_action
                # return self.state_array(self.state), self.REWARDS['step'], self.steps >= self.max_steps, {}

        if agent in costly:
            rew = self.REWARDS['costly step']
        else:
            rew = self.REWARDS['step']

        done = False
        if action == 0:  # MOVE
            if (agent + 1) % self.world_dim != 0:
                agent += 1
        elif action == 1:
            if agent + self.world_dim < self.world_dim * self.world_dim:
                agent += self.world_dim
        elif action == 2:
            if agent % self.world_dim != 0:
                agent -= 1
        elif action == 3:
            if agent >= self.world_dim:
                agent -= self.world_dim
        elif action == 4:
            if agent == goal:
                done = True
                rew = self.REWARDS['goal']
            else:
                rew = self.REWARDS['wrong_goal']

        done = done or (self.steps >= self.max_steps)

        self.state['agent'] = agent

        self.steps += 1

        return self.state_array(self.state), rew, done, {}

    def close(self):
        pass

    def render(self):
        self.render_state(self.state)

    def reset(self):
        self.steps = 0

        agent = random.choice(self.AGENT_START_STATES)
        goal = random.choice(self.GOAL_STATES)

        self.state = {
            'agent': agent,
            'goal': goal,
            'frozen': self.FROZEN_SQUARES,
            'costly': self.COSTLY_SQUARES
        }

        return self.state_array(self.state)

    def render_state(self, x):
        ''' Renders single state x '''
        rendering = '---------------'
        print('STATE = {}'.format(x))

        frozen = self.FROZEN_SQUARES
        costly = self.COSTLY_SQUARES
        agent = x[0]
        goal = x[1]

        for i in range(self.world_dim * self.world_dim):
            if i % self.world_dim == 0:
                rendering += '\n'

            if i == agent:
                rendering += ' A '
            elif i in frozen:
                rendering += ' F '
            elif i in costly:
                rendering += ' C '
            elif i == goal:
                rendering += ' G '
            else:
                rendering += ' - '

        rendering += '\n'
        rendering += '---------------'
        print(rendering)

    def realistic(self, x):
        ''' Returns a boolean indicating if x is a valid state in the environment (e.g. chess state without kings is not valid)'''
        return True

    def actionable(self, x, fact):
        ''' Returns a boolean indicating if all immutable features remain unchanged between x and fact states'''
        return True

    def get_actions(self, x):
        ''' Returns a list of actions available in state x'''
        return list(self.ACTIONS.values())

    def set_state(self, x):
        ''' Changes the environment's current state to x '''
        self.state = {}
        self.state['agent'] = x[0]
        self.state['goal'] = x[1]

        self.state['frozen'] = self.FROZEN_SQUARES
        self.state['costly'] = self.COSTLY_SQUARES

        self.steps = 0

    def check_done(self, x):
        ''' Returns a boolean indicating if x is a terminal state in the environment'''
        return False

    def equal_states(self, x1, x2):
        ''' Returns a boolean indicating if x1 and x2 are the same state'''
        return list(x1) == list(x2)

    def writable_state(self, x):
        ''' Returns a string with all state information to be used for writing results'''
        return 'Agent: {} Goal: {}'.format(x[0], x[1])

    def generate_state_from_json(self, json_dict):
        agent = json_dict['agent']
        goal = json_dict['goal']

        state = {
            'agent': agent,
            'goal': goal,
            'frozen': self.FROZEN_SQUARES,
            'costly': self.COSTLY_SQUARES
        }

        return self.state_array(state)

    def state_array(self, x):
        array_state = []
        array_state.append(x['agent'])
        array_state.append(x['goal'])

        for f in self.FROZEN_SQUARES:
            array_state.append(f)

        for c in self.COSTLY_SQUARES:
            array_state.append(c)

        return np.array(array_state)