from src.optimization.algs.heuristic_ts_alg import HeuristicTSAlgorithm
from src.optimization.search.tree_search import TreeSearch


class HeuristicTreeSearch(TreeSearch):

    def __init__(self, env, bb_model, obj, params):
        alg = HeuristicTSAlgorithm(env, bb_model, obj, params)
        super(HeuristicTreeSearch, self).__init__(env, bb_model, obj, params, alg)


