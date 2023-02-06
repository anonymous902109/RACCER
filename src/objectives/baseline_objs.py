from math import sqrt

import torch


class BaselineObjectives:

    def __init__(self, env, bb_model, enc, enc_dataset, n_var):
        self.env = env
        self.bb_model = bb_model
        self.enc = enc
        self.n_var = n_var
        self.enc_dataset = enc_dataset

        self.lmbdas = {'proximity': -1,
                       'sparsity': -1,
                       'dmc': -1,
                       'validity': 0,
                       'realistic': -1,
                       'actionable': -1}  # because BO_MCTS maximizes value

        self.num_objectives = len(self.lmbdas)

    def get_ind_rews(self, fact, cf, target_action, actions=None, cummulative_reward=None):
        objectives = self.get_objectives(fact)
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

    def get_collapsed_obj(self, fact):
        objs = self.get_objectives(fact)

        collapsed_obj = lambda x: sum([objs[k](x) * self.lmbdas[k] for k in objs.keys()])

        return collapsed_obj

    def get_reward(self, fact, cf, target_action, actions=None, cummulative_reward=0):
        objectives = self.get_objectives(fact)
        contraints = self.get_constraints(fact, target_action)

        final_rew = 0.0

        for o_name, o_formula in objectives.items():
            final_rew += self.lmbdas[o_name] * o_formula(cf)

        for c_name, c_formula in contraints.items():
            final_rew += self.lmbdas[c_name] * c_formula(cf)

        return final_rew

    def get_objectives(self, fact):
        self.max_proximity = self.enc.max_diff(fact)
        return {
            'proximity': lambda x: self.proximity(x, fact),
            'sparsity': lambda x: self.sparsity(x, fact),
            'dmc': lambda x: self.data_manifold_closeness(x, self.enc_dataset)
        }

    def get_constraints(self, fact, target):
        return {
            'validity': lambda x: int(self.bb_model.predict(x) != target),  # validity
            'realistic': lambda x: 1 - self.env.realistic(x),  # realistic
            'actionable': lambda x: 1 - self.env.actionable(x, fact)  # actionable
        }

    def proximity(self, x, fact):
        x_tensor = torch.tensor(x).squeeze()
        enc_x = self.enc.encode(x_tensor)

        fact_tensor = torch.tensor(fact).squeeze()
        enc_fact = self.enc.encode(fact_tensor)

        diff = abs(torch.subtract(enc_x, enc_fact))

        return sum(diff).item() / self.max_proximity

    def sparsity(self, x, fact):
        return (sum(fact[:-1] != x[:-1]) * 1.0).item() # last feature is player in chess or chopping in gridworld -- not important for sparsity

    def data_manifold_closeness(self, x, data):
        x_tensor = torch.tensor(x).squeeze()
        enc_x = self.enc.encode(x_tensor)
        recon = self.enc.decode(enc_x)

        diff = sqrt(sum((recon - x_tensor)**2))

        return diff