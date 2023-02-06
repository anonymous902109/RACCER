import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from src.models.counterfactual import CF


class GeneticBaseline:

    def __init__(self, env, bb_model, dataset, obj, params):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim

        self.obj = obj

        self.sample_size = params['gen_sample_size']
        self.n_iter = params['gen_iter']

    def generate_counterfactuals(self, fact, target, nbhd):
        print('Generating counterfactuals...')
        pop_size = self.sample_size

        X = np.tile(fact, (pop_size, 1))

        objs = self.obj.get_collapsed_obj(fact=fact)
        constr_ieq = list(self.obj.get_constraints(fact, target).values())

        problem = FunctionalProblem(self.n_var,
                                    objs,
                                    constr_ieq=constr_ieq,
                                    xl=self.env.lows,
                                    xu=self.env.highs)

        algorithm = GA(pop_size=pop_size,
                       sampling=X,
                       crossover=SBX(prob=1, vtype=int, repair=RoundingRepair()),
                       mutation=PM(prob=1, vtype=int, repair=RoundingRepair()),
                       eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       ('n_gen', self.n_iter),
                       seed=1,
                       verbose=True)

        solutions = res.pop.get('X')
        F = res.pop.get('F')
        G = res.pop.get('G')

        valid_cfs = []
        cf_values = []
        for i, s in enumerate(solutions):
            f = F[i]
            g = G[i]

            # if cf obeys the constraints
            if sum(g) == 0:
                cf_values.append(f)
                valid_cfs.append(s)

        if len(valid_cfs) == 0:
            return None

        print('Found {} cfs'.format(len(valid_cfs)))
        best_index = np.argmax(np.array(cf_values))
        best_cf = valid_cfs[best_index]

        nn = nbhd.find(best_cf)
        if nn is not None:
            cf = CF(best_cf, True, nn.actions, nn.cumulative_rew, F[best_index], 0, 0)
        else:
            cf = CF(best_cf, True, [0] * 10, -1 * 10, F[best_index], 0, 0)

        return cf




