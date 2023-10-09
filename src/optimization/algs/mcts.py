import operator

import numpy as np
from src.optimization.algs.tree_node import TreeNode


class MCTS:

    def __init__(self, env, bb_model, obj, params):
        self.env = env
        self.bb_model = bb_model
        self.obj = obj

        self.n_expand = params['ts_n_expand']
        self.n_sim = params['n_sim']
        self.max_level = params['max_level']
        self.n_iter = params['ts_n_iter']
        self.c = params['c']

        self.tree_size = 0
        self.cfs = []

    def search(self, init_state, fact, target_action):
        self.root = TreeNode(init_state, None, None, 0, self.env, self.bb_model, self.obj, fact, target_action)
        self.cfs = []

        i = 0
        while i < self.n_iter:
            i += 1

            node = self.select(self.root)

            if (not node.is_terminal()) and (node.level < self.max_level):
                new_nodes, exp_actions = self.expand(node)

                for j in range(len(exp_actions)):
                    e = self.simulate(node, exp_actions[j])

                    any_child = np.random.choice(node.children[exp_actions[j]])
                    self.backpropagate(any_child, e)

                self.store_cfs(new_nodes, target_action)

        return self.cfs

    def select(self, root):
        node = root

        node.n_visits += 1

        while (not node.is_terminal()) and (len(node.children) > 0):
            action_vals = {}
            exploit_vals = {}

            for a in node.available_actions():
                if len(node.N_a) == 0:
                    n_a = 0
                else:
                    n_a = node.N_a[a]

                if len(node.N_bp) == 0:
                    n_backprop = 0
                else:
                    n_backprop = node.N_bp[a]

                rew = node.rewards[a]

                action_value, exploit_value = self.ucb(rew, n_backprop, n_a, node.n_visits, self.c)

                action_vals[a] = action_value
                exploit_vals[a] = exploit_value

            next_action = max(action_vals.items(), key=operator.itemgetter(1))[0]

            try:
                node.N_a[next_action] += 1
            except KeyError:
                node.N_a[next_action] = 1

            node.n_visits += 1

            child = np.random.choice(node.children[next_action])

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

                node.expanded_actions.append(action)

                # add new nodes to the tree
                for i, ns in enumerate(new_states):
                    if ns.is_valid():  # checks if new node is a valid game state
                        try:
                            node.children[action].append(ns)
                        except KeyError:
                            node.children[action] = [ns]

                        nns.append(ns)

                        self.tree_size += 1

        return nns, node.expanded_actions

    def simulate(self, node, action):
        node = node.clone()
        evals = {}

        for i in range(self.n_sim):
            l = len(node.prev_actions)
            start_node = node.clone()

            next_nodes, _ = start_node.take_action(action, n_expand=1, expand=False)
            start_node = next_nodes[0]

            while (not start_node.is_terminal()) and (l < self.max_level - 1):  # minus 1 because one action is already added beyond the while loop
                l += 1

                rand_action = np.random.choice(start_node.available_actions())
                next_nodes, _ = start_node.take_action(rand_action, n_expand=1, expand=False)

                start_node = next_nodes[0]

            e = start_node.get_reward_dict()
            evaluation = e

            if len(evals):
                for k, v in evaluation.items():
                    evals.update({k: v + evals[k]})
            else:
                evals = evaluation

        # average evaluations
        for k, v in evals.items():
            evals.update({k: v / (self.n_sim * 1.0)})

        return evals

    def backpropagate(self, node, eval):
        while node.parent is not None:
            prev_action = node.prev_actions[-1]

            if len(node.parent.rewards) and prev_action in list(node.parent.rewards.keys()):
                for k, v in node.parent.rewards[prev_action].items():
                    node.parent.rewards[prev_action][k] = v + eval[k]
            else:
                node.parent.rewards[prev_action] = eval

            if len(node.parent.N_bp) == 0 or (prev_action not in list(node.parent.N_bp.keys())):
                node.parent.N_bp[prev_action] = 1
            else:
                node.parent.N_bp[prev_action] += 1

            node = node.parent

    def ucb(self, rewards, n_bp, n_a, n_visits, c):
        ucb = []
        just_exploit = []

        for r_key, r_value in rewards.items():
            exploit = r_value / n_bp

            explore = np.sqrt(
                (4.0 * np.log(n_visits) + np.log(len(rewards)))
                / (2.0 * n_a)
            )

            val = exploit + c * explore
            ucb.append(val)
            just_exploit.append(exploit)

        return ucb, just_exploit

    def store_cfs(self, new_nodes, target_action):
        for nn in new_nodes:
            if self.bb_model.predict(nn.state) == target_action:
                exists = False

                for cf in self.cfs:
                    if cf[0] == nn.prev_actions:
                        exists = True
                        break

                if not exists:
                    self.cfs.append((nn.prev_actions, nn.state, nn.rewards, nn.rank_value))
