from src.baselines.abs_baseline import AbstractBaseline
from src.models.counterfactual import CF

from src.optimization.objs.fid_obj import FidObj


import numpy as np

from src.optimization.search.mc_tree_search import MCTreeSearch


class MCTSRACCER(AbstractBaseline):

    def __init__(self, env, bb_model, params):
        self.obj = FidObj(env, bb_model, params)
        self.optim = MCTreeSearch(env, bb_model, self.obj, params)

        self.objectives = ['fidelity', 'reachability', 'stochastic_validity']

        super(MCTSRACCER, self).__init__()

    def generate_counterfactuals(self, fact, target):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target):
        ''' Returns all cfs found in the tree '''
        cfs = self.optim.alg.search(init_state=fact, fact=fact, target_action=target)

        if len(cfs):
            cfs = [CF(cf[0], cf[1], cf[2], cf[3]) for cf in cfs]
            cf_values = [cf.value for cf in cfs]

            best_value = max(cf_values)
            best_cfs = [cf for cf in cfs if cf.value == best_value]

            best_cf = self.choose_closest(best_cfs, fact)
            return best_cf
        else:
            return None

    def choose_closest(self, cfs, fact):
        diffs = []
        for c in cfs:
            d = sum(c.cf != fact)
            diffs.append(d)

        min_diff_index = np.argmax(diffs)
        return cfs[min_diff_index]