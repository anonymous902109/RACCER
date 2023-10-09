from src.baselines.abs_baseline import AbstractBaseline
from src.models.counterfactual import CF
from src.optimization.objs.rl_objs import RLObjs
from src.optimization.search.heuristic_tree_search import HeuristicTreeSearch


class RACCER(AbstractBaseline):

    def __init__(self, env, bb_model, params):
        self.obj = RLObjs(env, bb_model, params, max_actions=params['max_actions'])
        self.optim = HeuristicTreeSearch(env, bb_model, self.obj, params)

        self.pareto_metrics = ['cost', 'reachability', 'validity']

        super(RACCER, self).__init__()

    def generate_counterfactuals(self, fact, target):
        return self.get_all_cfs(fact, target)

    def get_all_cfs(self, fact, target):
        ''' Returns all cfs found in the tree '''
        self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        all_nodes = self.optim.traverse(self.optim.alg.root)

        potential_cf = [CF(n.prev_actions, n.rewards) for n in all_nodes if n.is_terminal()]

        pareto_front = self.get_pareto_front(potential_cf)

        return pareto_front
