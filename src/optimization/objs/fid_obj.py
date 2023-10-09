from src.optimization.objs.abs_obj import AbstractObj


class FidObj(AbstractObj):

    def __init__(self, env, bb_model, params):
        self.bb_model = bb_model
        max_actions = params['max_actions']
        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']
        super(FidObj, self).__init__(env, bb_model, params, max_actions)

    def get_objectives(self, fact, cf, actions, target_action):
        stochasticity, validity, cost, fidelity = self.calculate_stochastic_rewards(fact, actions, target_action, self.bb_model)

        reachability = self.reachability(actions)

        return {
            'fidelity': fidelity,
            'reachability': reachability,
            'stochastic_validity': validity
        }