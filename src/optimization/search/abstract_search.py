class AbstractSearch():
    ''' Abstract class describing a search algorithm '''

    def __init__(self, env, bb_model, obj, params):
        self.env = env
        self.bb_model = bb_model
        self.obj = obj
        self.params = params

    def generate_counterfactuals(self, fact, target):
        ''' Generates a counterfactual for given fact and target action '''
        return None