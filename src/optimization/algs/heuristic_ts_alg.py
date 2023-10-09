import numpy as np
import math

from src.optimization.algs.tree_node import TreeNode


class HeuristicTSAlgorithm:

    def __init__(self, env, bb_model, obj, params):
        self.env = env
        self.bb_model = bb_model
        self.obj = obj

        self.n_expand = params['ts_n_expand']
        self.max_level = params['max_level']
        self.n_iter = params['ts_n_iter']
        self.c = params['c']

        self.tree_size = 0

    def search(self, init_state, fact, target_action):
        self.root = TreeNode(init_state, None, None, 0, self.env, self.bb_model, self.obj, fact, target_action)
        self.cfs = []

        i = 0
        while i < self.n_iter:
            i += 1

            node = self.select(self.root)

            if (not node.is_terminal()) and (node.level < self.max_level):
                new_nodes, action = self.expand(node)

                for c in new_nodes:
                    c.rank_value = c.get_reward()
                    c.rewards = c.get_reward_dict()

                    if self.bb_model.predict(c.state) == target_action:
                        if self.env.realistic(c.state) and self.env.actionable(c.state, fact):
                            self.cfs.append((c.prev_actions, c.state, c.rewards, c.rank_value))

                if len(new_nodes):
                    self.backpropagate(new_nodes[0].parent)

        return self.cfs

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

    def backpropagate(self, node):
        while node is not None:
            node.n_visits += 1

            for a in node.expanded_actions:
                try:
                    node.Q_values[a] = np.mean([n.rank_value for n in node.children[a]])
                except KeyError:
                    node.Q_values[a] = -1000

            node = node.parent