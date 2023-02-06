import copy
import random

import gym
import numpy as np


class Gridworld(gym.Env):

    def __init__(self):
        self.world_dim = 5

        self.state_dim = self.world_dim ** 2 + 1

        self.chopping = 0
        self.max_chopping = 5

        self.step_pen = -1
        self.goal_rew = 10

        self.max_steps = 100
        self.steps = 0

        self.lows = np.array([0]*self.state_dim)
        self.highs = np.array([5] * 25 + [self.max_chopping])
        self.observation_space = gym.spaces.Box(self.lows, self.highs, shape=(26, ))
        self.action_space = gym.spaces.Discrete(6)

        self.state = np.zeros((self.state_dim, ))

        self.num_trees = 2

        self.ACTIONS = {'RIGHT': 0, 'DOWN': 1, 'LEFT': 2, 'UP': 3, 'CHOP': 4, 'SHOOT': 5}
        self.OBJECTS = {'AGENT': 1, 'MONSTER': 2, 'TREE': 3, 'KILLED_MONSTER': -1}

        self.TREE_TYPES = {3: 5, 4: 3, 5: 1}
        self.FERTILITY = {2: 0.1, 7: 0.5, 12: 0.9, 17: 0.5, 22: 0.1}
        self.TREE_POS_TYPES = {2: 5, 7: 4, 12: 3, 17: 4, 22: 5}
        self.TREE_POS = [2, 7, 12, 17, 22]

    def step(self, action):
        if isinstance(action, str):
            action = self.ACTIONS[action]

        new_state, done, rew = self.get_new_state(self.state, action)

        self.state = new_state
        self.steps += 1

        return new_state.flatten(), rew, done, {}

    def create_state(self, agent, monster, trees, chopping, chopped_trees=[], killed_monster=False):
        state = [0.0] * self.state_dim
        state[-1] = chopping
        state[agent] = self.OBJECTS['AGENT']

        if killed_monster:
            state[monster] = self.OBJECTS['KILLED_MONSTER']
        else:
            state[monster] = self.OBJECTS['MONSTER']

        for t in trees:
            t_pos, t_type = tuple(t.items())[0]
            if isinstance(t_pos, str):
                t_pos = int(t_pos)

            if t_pos not in chopped_trees:
                state[t_pos] = t_type

        return np.array(state)

    def get_new_state(self, state, action):
        agents, monsters, trees = self.get_objects(state)
        agent = agents[0]
        monster = monsters[0]

        facing_monster = self.facing_obstacle(agent, [monster], action)
        facing_tree = self.facing_obstacle(agent, [list(t.keys())[0] for t in trees], action)

        chopped_trees = []

        if action == 0:  # MOVE
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if (agent + 1) % self.world_dim != 0:
                    agent += 1
        elif action == 1:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent + self.world_dim < self.world_dim * self.world_dim:
                    agent += self.world_dim
        elif action == 2:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent % self.world_dim != 0:
                    agent -= 1
        elif action == 3:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent >= self.world_dim:
                    agent -= self.world_dim
        elif action == 4:  # CHOP
            near_trees = self.get_neighboring_trees(agent, trees)
            if len(near_trees):
                t_pos, t_type = tuple(near_trees[0].items())[0]  # start with first tree
                self.chopping += 1
                if self.chopping >= self.TREE_TYPES[t_type]:
                    chopped_trees.append(t_pos)
                    self.chopping = 0

        elif action == 5:  # SHOOT
            self.chopping = 0
            if (int(agent / self.world_dim) == int(monster / self.world_dim)) or (agent % self.world_dim == monster % self.world_dim):
                free = self.check_if_path_free(agent, monster, trees)
                if free:
                    new_array = self.create_state(agent, monster, trees, self.chopping, killed_monster=True)
                    return new_array, True, self.goal_rew

        # regrow trees in the middle column
        new_trees = self.regrow(trees, agent, monster, chopped_trees)
        trees += new_trees

        new_state = self.create_state(agent, monster, trees, self.chopping, chopped_trees)

        self.state = new_state

        return new_state, self.steps >= self.max_steps, self.step_pen

    def regrow(self, trees, agent, monster, chopped_trees):
        tree_occupied = [list(t.keys())[0] for t in trees]
        free_squares = [s for s in self.TREE_POS if (s not in tree_occupied) and (s not in chopped_trees) and (s != agent) and (s != monster)]
        if len(free_squares) == 0:
            return []

        new_trees = []
        for i in free_squares:
            p = self.FERTILITY[i]
            regrow_i = random.choices([0, 1], weights=[1-p, p])[0]
            if regrow_i == 1:
                tree_type = self.TREE_POS_TYPES[i]
                new_trees.append({i: tree_type})

        return new_trees

    def get_neighboring_trees(self, agent, trees):
        nts = []
        for t in trees:
            t_pos, t_type = tuple(t.items())[0]
            if self.next_to_obstacle(agent, t_pos):
                nts.append(t)

        return nts

    def facing_obstacle(self, agent, obstacles, action):
        for o in obstacles:
            if ((agent + 1 == o) and ((agent + 1) % self.world_dim != 0) and  action == self.ACTIONS['RIGHT']) \
                    or (agent + self.world_dim == o and action == self.ACTIONS['DOWN']) \
                    or ((agent - 1 == o) and (agent % self.world_dim != 0) and action == self.ACTIONS['LEFT']) \
                    or (agent - self.world_dim == o and action == self.ACTIONS['UP']):
                return True

        return False

    def next_to_obstacle(self, agent, obstacle):
        if ((agent + 1 == obstacle) and ((agent + 1) % self.world_dim != 0)) \
                or (agent + self.world_dim == obstacle) \
                or ((agent - 1 == obstacle) and (agent % self.world_dim != 0)) \
                or (agent - self.world_dim == obstacle):
            return True

        return False

    def check_if_path_free(self, agent, monster, trees):
        if int(agent / self.world_dim) == int(monster / self.world_dim):
            for t in trees:
                t_pos, t_type = tuple(t.items())[0]
                if t_pos > min([agent, monster]) and t_pos < max([agent, monster]):
                    return False

        if (agent % self.world_dim == monster % self.world_dim):
            for t in trees:
                t_pos, t_type = tuple(t.items())[0]
                if t_pos % self.world_dim == monster % self.world_dim and t_pos > min([agent, monster]) and t_pos < max([agent, monster]):
                    return False

        return True

    def reset(self):
        monster = random.randint(0, self.world_dim * self.world_dim - 1)
        agent = random.randint(0, self.world_dim * self.world_dim - 1)

        while agent % 5 > 1:
            agent = random.randint(0, self.world_dim * self.world_dim - 1)

        while monster % 5 < 3:
            monster = random.randint(0, self.world_dim * self.world_dim - 1)

        tree_wall = np.array(self.TREE_POS)
        tree_pos = np.random.uniform(0, 1, 5) > 0.2
        tree_pos = tree_wall[tree_pos]
        trees = []
        for t in tree_pos:
            tree_type = self.TREE_POS_TYPES[t]
            trees.append({t: tree_type})

        self.chopping = 0

        self.state = self.create_state(agent, monster, trees, self.chopping)

        self.steps = 0
        return self.state.flatten()

    def close(self):
        pass

    def render(self):
        self.render_state(self.state)

    def render_state(self, state):
        if isinstance(state, list):
            state = np.array(state)

        agents, monsters, trees = self.get_objects(state)

        rendering = '---------------'
        print('STATE = {}'.format(state))

        for i in range(self.world_dim * self.world_dim):
            if i % self.world_dim == 0:
                rendering += '\n'

            if i in agents:
                rendering += ' A '
            elif i in monsters:
                rendering += ' M '
            else:
                tree_found = False
                for t in trees:
                    t_pos, t_type = tuple(t.items())[0]
                    if i == t_pos:
                        rendering += ' T{} '.format(t_type)
                        tree_found = True

                if not tree_found:
                    rendering += ' - '

        rendering += '\n'
        rendering += '---------------'
        print(rendering)

    def get_objects(self, x):
        x = np.array(x).squeeze()

        agent = list(np.where(x[0:self.world_dim * self.world_dim] == self.OBJECTS['AGENT'])[0])
        monster = list(np.where(x[0:self.world_dim * self.world_dim] == self.OBJECTS['MONSTER'])[0])

        trees = []
        for t_type in self.TREE_TYPES.keys():
            tree_type_list = list(np.where(x[0:self.world_dim * self.world_dim] == t_type)[0])
            for t_pos in tree_type_list:
                trees.append({t_pos: t_type})

        return agent, monster, trees

    def realistic(self, x):
        agent, monster, trees = self.get_objects(x)

        total_trees = len(trees)

        t_pos = [list(i.keys())[0] for i in trees]
        t_types = [list(i.values())[0] for i in trees]

        for i, t in enumerate(t_pos):
            if t not in self.TREE_POS:
                return False

            if t_types[i] != self.TREE_POS_TYPES[t]:
                return False

        if len(agent) != 1:
            return False
        if len(monster) != 1:
            return False
        if total_trees > 5:
            return False

        return True

    def actionable(self, x, fact):
        monster = list(np.where(fact == self.OBJECTS['MONSTER'])[0])

        if len(monster) != 1:
            return False

        return abs(x[monster] == self.OBJECTS['MONSTER']).item()

    def generate_state_from_json(self, json_dict):
        agent = json_dict['agent']
        monster = json_dict['monster']
        trees = json_dict['trees']

        state = self.create_state(agent, monster, trees, chopping=0)

        return state

    def get_actions(self, state):
        return np.arange(self.action_space.n)

    def set_state(self, state):
        self.state = copy.deepcopy(state)
        self.chopping = self.state[-1]

    def check_done(self, state):
        killed_monster = list(np.where(state[0:self.world_dim * self.world_dim] == self.OBJECTS['KILLED_MONSTER'])[0])

        if len(killed_monster):
            return True

        return False

    def equal_states(self, s1, s2):
        return sum(s1 != s2) == 0

    def writable_state(self, s):
        agent, monster, trees = self.get_objects(s)
        ws = 'Agent: {} Monster: {} Trees: {}'.format(agent, monster, trees)
        return ws


