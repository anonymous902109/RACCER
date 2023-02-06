import numpy as np
from src.models.counterfactual import CF
from src.optimization.mcts import MCTS


class MCTSSearch:

    def __init__(self, env, bb_model, dataset, obj, params, c):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim
        self.obj = obj

        self.n_iter = params['ts_n_iter']
        self.n_expand = params['ts_n_expand']

        self.c = c

    def generate_counterfactuals(self, fact, target, nbhd=None):
        mcts_solver = MCTS(self.env, self.bb_model, self.obj, fact, target, c=self.c, max_level=5, n_expand=self.n_expand)
        found = False

        tree_size, time = mcts_solver.search(init_state=fact, num_iter=self.n_iter)

        all_nodes = self.traverse(mcts_solver.root)

        potential_cf = [CF(n.state, True, n.prev_actions, n.cumulative_reward, n.value, tree_size, time)
                        for n in all_nodes if n.is_terminal()]

        # return only the best one
        print('Found {} counterfactuals'.format(len(potential_cf)))
        if len(potential_cf):
            best_cf_ind = np.argmax([cf.value for cf in potential_cf])
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