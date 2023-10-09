import numpy as np
from src.models.counterfactual import CF


class TreeSearch:

    def __init__(self, env, bb_model, obj, params, alg):
        self.env = env
        self.bb_model = bb_model
        self.n_var = env.state_dim
        self.obj = obj
        self.alg = alg

    def generate_counterfactuals(self, fact, target):
        return self.get_best_counterfactual(fact, target)

    def get_best_counterfactual(self, fact, target):
        tree_size, time = self.alg.search(init_state=fact, fact=fact, target_action=target)

        all_nodes = self.traverse(self.alg.root)

        potential_cf = [CF(n.prev_actions, n.rewards, n.rank_value)
                        for n in all_nodes if n.is_terminal()]

        # return only the best one
        if len(potential_cf):
            best_cf_ind = np.argmax([cf.rank_value for cf in potential_cf])
            try:
                best_cf = potential_cf[best_cf_ind]
            except IndexError:
                return None
        else:
            return None

        return best_cf

    def traverse(self, root, nodes=None):
        ''' Returns all nodes in the tree '''
        if nodes is None:
            nodes = set()

        nodes.add(root)

        if root.children is not None and len(root.children):
            children = []
            for action in root.children.keys():
                children += root.children[action]

            for c in children:
                self.traverse(c, nodes)

        return nodes