class RLObjs:

    def __init__(self, env, bb_model, params, max_actions):
        self.env = env
        self.bb_model = bb_model
        self.max_actions = max_actions

        self.lmbdas = {'cost': -1,
                       'reachability': -1,
                       'stochasticity': -1,
                       'validity': 0,
                       'realistic': -0,
                       'actionable': -0}  # because BO_MCTS maximizes value

        self.num_objectives = len(self.lmbdas)

        self.n_sim = params['n_sim']

    def get_ind_rews(self, fact, cf, target_action, actions, cummulative_reward):
        objectives = self.get_objectives(fact, actions, cummulative_reward, target_action)
        contraints = self.get_constraints(fact, target_action)

        rewards = {}
        total_rew = 0.0

        for o_name, o_formula in objectives.items():
            rewards[o_name] = self.lmbdas[o_name] * o_formula(cf)
            total_rew += rewards[o_name]

        for c_name, c_formula in contraints.items():
            rewards[c_name] = self.lmbdas[c_name] * c_formula(cf)
            total_rew += rewards[c_name]

        return rewards, total_rew

    def get_reward(self, fact, cf, target_action, actions, cummulative_rew):
        objectives = self.get_objectives(fact, actions, cummulative_rew, target_action)
        contraints = self.get_constraints(fact, target_action)

        final_rew = 0.0

        for o_name, o_formula in objectives.items():
            final_rew += self.lmbdas[o_name] * o_formula(cf)

        for c_name, c_formula in contraints.items():
            final_rew += self.lmbdas[c_name] * c_formula(cf)

        return final_rew

    def get_objectives(self, fact, actions, cummulative_rew, target_action):
        return {
            'cost': lambda x: self.cost(x, fact, actions, cummulative_rew),
            'reachability': lambda x: self.reachability(x, fact, actions),
            'stochasticity': lambda x: self.stochasticity(x, fact, actions, target_action)
        }

    def get_constraints(self, fact, target):
        return {
            'validity': lambda x: self.bb_model.predict(x) != target,  # validity
            'realistic': lambda x: 1 - self.env.realistic(x),  # realistic
            'actionable': lambda x: 1 - self.env.actionable(x, fact)  # actionable
        }

    def cost(self, x, fact, actions, cummulative_rew):
        if not len(actions):
            return 0
        return (cummulative_rew*1.0) / (len(actions))  # max penalty is -1, across the path -1 * len(actions)

    def reachability(self, x, fact, actions):
        return len(actions) * 1.0 / self.max_actions

    def stochasticity(self, x, fact, actions, target_action):
        # run simulations from fact with actions
        n_sim = self.n_sim

        cnt = 0
        for s in range(n_sim):
            self.env.reset()
            self.env.set_state(fact)

            if len(actions) == 0:
                return 1

            done = False
            early_break = False
            available_actions = self.env.get_actions(fact)
            for a in actions:
                if done or (a not in available_actions) or (len(available_actions) == 0):
                    early_break = True
                    break

                obs, rew, done, _ = self.env.step(a)

                available_actions = self.env.get_actions(obs)

            if not early_break:
                # count how many times  ends up in x
                if self.bb_model.predict(obs) == target_action:
                    cnt += 1

        # percentage of ending up in x
        return 1 - ((cnt*1.0) / n_sim)
