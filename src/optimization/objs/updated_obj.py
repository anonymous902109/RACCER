from src.optimization.objs.abs_obj import AbstractObj


class UpdatedObj(AbstractObj):

    def __init__(self, env, bb_model, params, max_actions):
        super(UpdatedObj, self).__init__(env, bb_model, params, max_actions)

    def get_ind_unweighted_rews(self, fact, cf, target_action, actions, cummulative_reward):
        objectives = self.get_objectives(fact, actions, cummulative_reward, target_action)

        rewards = {}
        total_rew = 0.0

        for o_name, o_formula in objectives.items():
            rewards[o_name] = -o_formula
            total_rew += -rewards[o_name]

        return rewards, total_rew


