class NBHDNode:

    def __init__(self, state, parent, action, rew, env):
        self.state = state
        self.parent = parent
        self.env = env

        self.actions = self.parent.actions + [action] if self.parent else []
        self.cumulative_rew = self.parent.cumulative_rew + rew if self.parent else 0
        self.level = self.parent.level + 1 if self.parent else 0

    def take_action(self, action):
        nns = []
        rewards = []

        for i in range(20):
            self.env.reset()
            self.env.set_state(self.state)

            obs, rew, done, _ = self.env.step(action)

            found = False
            for nn in nns:
                if self.env.equal_states(obs, nn.state):
                    found = True
                    break

            if not found and not done:
                nn = NBHDNode(obs, self, action, rew, self.env)
                nns.append(nn)
                rewards.append(rew)

        return nns, rewards


class NBHD:

    def __init__(self, env, fact, max_level):
        self.fact = fact
        self.env = env
        self.max_level = max_level

        self.build_tree(fact, max_level)

    def build_tree(self, root, max_level):
        print('Building tree with {} levels'.format(max_level))
        l = 0
        root_node = NBHDNode(root, None, None, None, self.env)
        self.tree = [root_node]

        while l < max_level:
            for node in self.tree:
                if node.level == l:
                    available_actions = self.env.get_actions(node.state)
                    for a in available_actions:
                        new_states, rewards = node.take_action(a)
                        for i, ns in enumerate(new_states):
                            self.tree.append(ns)

            l += 1

        print('Tree built!')

    def find(self, state):
        for node in self.tree:
            if self.env.equal_states(state, node.state):
                return node

        return None
