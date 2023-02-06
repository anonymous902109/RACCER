from os.path import exists

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.nbhd import NBHD



class Task:

    def __init__(self, task_name, env, bb_model, dataset, method, method_name, search_objs, eval_objs, eval_path, nbhd):
        self.task_name = task_name
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.method = method
        self.method_name = method_name
        self.eval_path = eval_path
        self.search_objs = search_objs
        self.eval_objs = eval_objs
        self.nbhd = nbhd

    def run_experiment(self, facts, targets=None):
        print('Running experiment for {} task with {}'.format(self.task_name, self.method_name))
        print('Finding counterfactuals for {} facts'.format(len(facts)))

        # get cfs for facts
        eval_dict = {}
        cnt = 0
        for i in tqdm(range(len(facts))):
            f = facts[i]

            if isinstance(f, dict):
                f = self.env.generate_state_from_json(f)

            if self.nbhd:
                nbhd = NBHD(self.env, f, max_level=0)
            else:
                nbhd = None

            if targets is None:
                ts = self.get_targets(f, self.env, self.bb_model)
            else:
                ts = [targets[i]]

            for t in ts:
                print('FACT: Target = {}'.format(t))
                self.env.render_state(f)

                cf = self.method.generate_counterfactuals(f, t, nbhd)

                if cf is None:
                    found = False
                    self.evaluate_cf(f, t, cf, found)
                    continue
                else:
                    found = True
                    self.evaluate_cf(f, t, cf, found)

                    print('CF:')
                    self.env.render_state(cf.cf_state)

                    cnt += 1

    def get_targets(self, f, env, bb_model):
        pred = bb_model.predict(f)
        available_actions = env.get_actions(f)
        targets = [a for a in available_actions if a != pred]

        return targets

    def evaluate_cf(self, f, t, cf, found):
        if not found:
            eval_obj_names = []

            for obj in self.eval_objs:
                eval_obj_names += list(obj.lmbdas.keys())

            ind_rew = [0] * len(eval_obj_names)
            df = pd.DataFrame([ind_rew], columns=eval_obj_names)

            df['total_reward'] = 0
            df['cf'] = 0

        else:
            rews = {}
            for obj in self.eval_objs:
                ind_rew, total_rew = obj.get_ind_rews(f, cf.cf_state, t, cf.actions, cf.cumulative_reward)
                ind_rew = {k: [v] for k, v in ind_rew.items()}
                rews.update(ind_rew)

            df = pd.DataFrame([rews])
            total_rew = self.search_objs.get_reward(f, cf.cf_state, t, cf.actions, cf.cumulative_reward)

            print(rews)
            df['total_reward'] = total_rew

            df['cf'] = self.env.writable_state(cf.cf_state)

        df['fact'] = list(np.tile(self.env.writable_state(f), (len(df), 1)))
        df['target'] = t
        df['found'] = found

        header = not exists(self.eval_path)
        df.to_csv(self.eval_path, mode='a', header=header)

        return ind_rew

    def true_path(self, env, model, start_state):
        env.reset()
        env.set_state(start_state)

        done = False
        true_path = []
        obs = start_state
        while not done:
            action = model.predict(obs)
            true_path.append(action)
            obs, rew, done, _ = env.step(action)

        return true_path