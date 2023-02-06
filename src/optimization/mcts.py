import copy

import numpy as np
import math


class MCTSNode:

    def __init__(self, state, parent, action, rew, env, bb_model,  obj, fact, target_action):
        self.state = state
        self.parent = parent
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.fact = fact
        self.target_action = target_action

        self.n_visits = 0
        self.N_a = {}

        self.children = {}

        self.prev_actions = self.parent.prev_actions + [action] if self.parent else []
        self.cumulative_reward = self.parent.cumulative_reward + rew if self.parent else 0

        self.expanded_actions = []
        self.Q_values = {}
        self.value = -1000

        self.level = self.parent.level + 1 if self.parent is not None else 0

    def available_actions(self):
        return self.env.get_actions(self.state)

    def is_terminal(self):
        return self.env.check_done(self.state) or self.bb_model.predict(self.state) == self.target_action

    def take_action(self, action, n_expand, expand=True):
        nns = []
        rewards = []
        s = n_expand if expand else 1

        for i in range(s):
            self.env.reset()
            self.env.set_state(self.state)

            obs, rew, done, _ = self.env.step(action)

            found = False
            for nn in nns:
                if self.env.equal_states(obs, nn.state):
                    found = True
                    break

            if not found:
                nn = MCTSNode(obs, self, action, rew, self.env, self.bb_model, self.obj, self.fact, self.target_action)
                nns.append(nn)
                rewards.append(rew)

        return nns, rewards

    def get_reward(self):
        return self.obj.get_reward(self.fact, self.state, self.target_action, self.prev_actions, self.cumulative_reward)

    def clone(self):
        clone = MCTSNode(self.state, None, None, None, self.env, self.bb_model,  self.obj, self.fact, self.target_action)
        clone.prev_actions = self.prev_actions
        clone.cumulative_reward = self.cumulative_reward

        return clone

    def is_valid(self):
        return self.env.realistic(self.state)

class MCTS:

    def __init__(self, env, bb_model, obj, fact, target_action, max_level=10, n_expand=20, c=None):
        self.max_level = max_level
        self.c = c if c is not None else 1 / math.sqrt(2)
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.fact = fact
        self.target_action = target_action
        self.n_expand = n_expand

        self.tree_size = 0

    def search(self, init_state, num_iter=200):
        self.root = MCTSNode(init_state, None, None, 0, self.env, self.bb_model, self.obj, self.fact, self.target_action)

        i = 0
        while i < num_iter:
            i += 1

            node = self.select(self.root)

            if (not node.is_terminal()) and (node.level < self.max_level):
                new_nodes, action = self.expand(node)

                for c in new_nodes:
                    c.value = c.get_reward()

                if len(new_nodes):
                    self.backpropagate(new_nodes[0].parent)

        return self.tree_size, 0

    def select(self, root):
        node = root

        while (not node.is_terminal()) and (len(node.children) > 0):
            action_vals = {}

            for a in node.available_actions():
                try:
                    n_a = node.N_a[a]
                    Q_val = node.Q_values[a]
                    action_value = Q_val + self.c * math.sqrt((math.log(node.n_visits) / n_a))
                    action_vals[a] = action_value

                except KeyError:
                    action_value = 0

            best_action = max(action_vals, key=action_vals.get)

            try:
                node.N_a[best_action] += 1
            except KeyError:
                node.N_a[best_action] = 1

            child = np.random.choice(node.children[best_action])

            node = child

        return node

    def expand(self, node):
        nns = []

        if len(node.available_actions()) == len(node.expanded_actions):
            return [], None

        if node.is_terminal():
            return [], None

        for action in node.available_actions():
            if action not in node.expanded_actions:

                new_states, new_rewards = node.take_action(action, n_expand=self.n_expand)

                try:
                    node.N_a[action] += 1
                except KeyError:
                    node.N_a[action] = 1

                node.expanded_actions.append(action)

                for i, ns in enumerate(new_states):
                    if ns.is_valid():
                        try:
                            node.children[action].append(ns)
                        except KeyError:
                            node.children[action] = [ns]

                        nns.append(ns)

                        self.tree_size += 1

        return nns, action

    def simulate(self, node):
        node = node.clone()
        n_sim = 1
        evals = []

        for i in range(n_sim):
            l = 0
            evaluation = 0.0
            start_node = node.clone()
            while (not start_node.is_terminal()) and (l < 5):
                l += 1

                rand_action = np.random.choice(start_node.available_actions())
                start_node = start_node.take_action(rand_action, n_expand=self.n_expand, expand=False)[0][0]

                e = start_node.get_reward()
                evaluation = e.item()

            evals.append(evaluation)

        return np.mean(evals)

    def backpropagate(self, node):
        while node is not None:
            node.n_visits += 1

            for a in node.expanded_actions:
                try:
                    node.Q_values[a] = np.mean([n.value for n in node.children[a]])
                except KeyError:
                    node.Q_values[a] = -1000

            node = node.parent