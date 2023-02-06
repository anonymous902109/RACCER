import pandas as pd
import numpy as np


def evaluate():
    task = 'chess'

    if task == 'gridworld':
        load = 500
        metrics = ['reachability', 'stochasticity', 'cost', 'proximity', 'sparsity', 'dmc']
    else:
        load = 63
        metrics = ['reachability', 'stochasticity', 'cost', 'proximity', 'sparsity', 'dmc', 'l_bo', 'l_rl']

    methods = ['BO_GEN', 'BO_MCTS', 'RL_MCTS']

    for m in methods:
        res = {}
        path = 'eval/{}/{}/rl_obj_results'.format(task, m)
        df = pd.read_csv(path, header=0)

        df = df.head(load)

        not_found = df[df['cf'] == '0']
        res['found'] = (load - len(not_found)) / load

        df = df[df['cf'] != '0']

        total_rew = list(np.zeros((len(df), )))

        for met in metrics:
            vals = df[met].values
            try:
                res[met] = np.mean(vals)
            except TypeError:
                vals = [float(v.split('[')[1].split(']')[0]) for v in vals if isinstance(v, str)]
                res[met] = np.mean(vals)

        print('Task = {}'.format(task))
        print('Method = {}'.format(m))
        print('Results = {}'.format(res))
        print('-----------------------------------------------------')

if __name__ == '__main__':
    evaluate()