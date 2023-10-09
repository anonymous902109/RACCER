from src.optimization.objs.abs_obj import AbstractObj


class GameObj(AbstractObj):

    def __init__(self, env, bb_model, params):
        self.bb_model = bb_model
        self.env = env
        self.objectives = ['realistic']

        max_actions = params['max_actions']
        super(GameObj, self).__init__(env, bb_model, params, max_actions)

    def get_objectives(self, fact, cf, actions, target_action):
        realistic = self.get_realistic(cf)

        return {
            'realistic': realistic
        }

    def get_realistic(self, cf):
        return self.env.realistic(cf)
