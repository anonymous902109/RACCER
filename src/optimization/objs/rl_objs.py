from src.optimization.objs.abs_obj import AbstractObj


class RLObjs(AbstractObj):

    def __init__(self, env, bb_model, params, max_actions):
        self.objectives = ['cost', 'reachability', 'sparsity']
        super(RLObjs, self).__init__(env, bb_model, params, max_actions)

    def get_objectives(self, fact, actions, cummulative_rew, target_action):
        stochasticity, validity, cost = self.calculate_stochastic_rewards(fact, actions, target_action)

        reachability = self.reachability(actions)

        return {
            'cost': cost,
            'reachability': reachability,
            'validity': validity
        }