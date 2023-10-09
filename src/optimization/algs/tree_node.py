class TreeNode:

    def __init__(self, state, parent, action, rew, env, bb_model, obj, fact, target_action):
        self.state = state
        self.parent = parent
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.fact = fact
        self.target_action = target_action

        self.n_visits = 1
        self.N_a = {a: 1 for a in self.env.get_actions(state)}
        self.N_bp = {}

        self.children = {}

        self.prev_actions = self.parent.prev_actions + [action] if self.parent else []
        self.cumulative_reward = self.parent.cumulative_reward + rew if self.parent else 0

        self.expanded_actions = []
        self.Q_values = {}
        self.rank_value = -1000

        self.rewards = {}

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
                nn = TreeNode(obs, self, action, rew, self.env, self.bb_model, self.obj, self.fact, self.target_action)
                nns.append(nn)
                rewards.append(rew)

        return nns, rewards

    def get_reward(self):
        return self.obj.get_reward(self.fact, self.state, self.target_action, self.prev_actions, self.cumulative_reward)

    def get_reward_dict(self):
        return self.obj.get_ind_unweighted_rews(self.fact, self.state, self.target_action, self.prev_actions, self.cumulative_reward)[0]

    def clone(self):
        clone = TreeNode(self.state, None, None, None, self.env, self.bb_model, self.obj, self.fact, self.target_action)
        clone.prev_actions = self.prev_actions
        clone.cumulative_reward = self.cumulative_reward

        return clone

    def is_valid(self):
        return self.env.realistic(self.state)

    def check_target_action(self):
        return self.obj.validity(self.fact, self.state, self.prev_actions, self.target_action)