from src.baselines.abs_baseline import AbstractBaseline
from src.optimization.objs.rl_objs import RLObjs
from src.optimization.search.heuristic_tree_search import HeuristicTreeSearch


class AutoRACCER(AbstractBaseline):

    def __init__(self, env, bb_model, params, weight_comb):
        self.obj = RLObjs(env, bb_model, params, max_actions=params['max_actions'])
        self.optim = HeuristicTreeSearch(env, bb_model, self.obj, params)
        self.weight_comb = weight_comb

        super(AutoRACCER, self).__init__()

    def generate_counterfactuals(self, fact, target):
        diverse_cf = []
        diverse_recourse = []
        for w in self.weight_comb:
            self.optim.obj.update_weights(w)
            cf = self.optim.get_best_counterfactual(fact, target)
            if cf is not None:
                if cf.recourse not in diverse_recourse:
                    diverse_cf.append(cf)
                    diverse_recourse.append(cf.recourse)

        pareto_cfs = self.get_pareto_front(diverse_cf)

        return pareto_cfs
