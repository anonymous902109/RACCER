from src.optimization.objs.abs_obj import AbstractObj


class SLObj(AbstractObj):

    def __init__(self, env, bb_model, params):
        self.bb_model = bb_model
        max_actions = params['max_actions']
        self.objectives = ['proximity', 'sparsity', 'validity']
        super(SLObj, self).__init__(env, bb_model, params, max_actions)

    def get_objectives(self, fact, cf, actions, target_action):
        proximity = self.get_proximity(fact, cf)
        sparsity = self.get_sparsity(fact, cf)
        validity = self.get_validity(cf, target_action)

        return {
            'proximity': proximity,
            'sparsity': sparsity,
            'validity': validity
        }

    def get_proximity(self, fact, cf):
        return 1 - (sum(abs(cf - fact)) / (self.env.max_feature*len(fact)))

    def get_sparsity(self, fact, cf):
        return 1 - (sum(fact == cf) / len(list(fact)))

    def get_validity(self, cf, t):
        return self.bb_model.predict(cf) == t