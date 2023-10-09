from src.optimization.algs.mcts import MCTS
from src.optimization.search.tree_search import TreeSearch


class MCTreeSearch(TreeSearch):

    def __init__(self, env, bb_model, obj, params):
        alg = MCTS(env, bb_model, obj, params)
        super(MCTreeSearch, self).__init__(env, bb_model, obj, params, alg)


