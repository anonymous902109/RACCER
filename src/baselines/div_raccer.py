from src.baselines.abs_baseline import AbstractBaseline
from src.models.counterfactual import CF
from src.optimization.objs.updated_obj import UpdatedObj
from src.optimization.search.pareto_mcts import ParetoMCTSSearch


class DivRACCER(AbstractBaseline):

    def __init__(self, env, bb_model, params):
        self.obj = UpdatedObj(env, bb_model, params, max_actions=params['max_actions'])
        self.optim = ParetoMCTSSearch(env, bb_model, self.obj, params)

        self.pareto_metrics = ['cost', 'reachability', 'validity']

        super(DivRACCER, self).__init__()

    def generate_counterfactuals(self, fact, target):
        return self.get_all_cfs(fact, target)

    def get_all_cfs(self, fact, target):
        ''' Returns all cfs found in the tree '''
        cfs = self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        all_cfs = [CF(cf[0], cf[1]) for cf in cfs]

        pareto_cfs = self.get_pareto_front(all_cfs)

        return pareto_cfs
