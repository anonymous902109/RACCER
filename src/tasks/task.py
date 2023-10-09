from os.path import exists

import pandas as pd
from tqdm import tqdm


class Task:

    def __init__(self, task_name, env, bb_model, method, method_name, eval_path, eval_objs, params):
        self.task_name = task_name
        self.env = env
        self.bb_model = bb_model
        self.method = method
        self.method_name = method_name
        self.eval_path = eval_path
        self.params = params

        self.eval_obj = eval_objs

    def run_experiment(self, facts, targets=None):
        print('Running experiment for {} task with {}'.format(self.task_name, self.method_name))
        print('Finding counterfactuals for {} facts'.format(len(facts)))

        # get cfs for facts
        cnt = 0

        for i in tqdm(range(len(facts))):  # for each fact
            f = facts[i]

            if isinstance(f, dict):
                f = self.env.generate_state_from_json(f)

            if targets is None:
                # if target is not given, go through all possible targets
                ts = self.get_targets(f, self.env, self.bb_model)
            else:
                ts = targets[i]

            for t in ts:
                cfs = self.method.generate_counterfactuals(f, t)

                if cfs is None:
                    found = False
                    self.evaluate_cf(f, t, cfs, found, cnt)
                    continue
                else:
                    if not isinstance(cfs, list):
                        cfs = [cfs]

                    for cf in cfs:
                        found = True
                        self.evaluate_cf(f, t, cf, found, cnt)

            cnt += 1

    def get_targets(self, f, env, bb_model):
        pred = bb_model.predict(f)
        available_actions = env.get_actions(f)
        targets = [a for a in available_actions if a != pred]

        return targets

    def evaluate_cf(self, f, t, cf, found, fact_id):
        eval_obj_names = []

        for eo in self.eval_obj:
            eval_obj_names += eo.objectives

        if not found:
            df_values = {e: -1 for e in eval_obj_names}

            df_values['recourse'] = 0
            df_values['cf_readable'] = 0
            df_values['cf'] = 0
        else:
            df_values = cf.reward_dict

            recourse = str(cf.recourse).split('[')[1].split(']')[0]
            if recourse == '':
                recourse = self.generate_recourse(f, cf.cf, self.env, self.params['max_actions'])
                if recourse != 0:
                    cf.recourse = recourse
                recourse = str(recourse)

            for eo in self.eval_obj:
                obj_vals = eo.get_objectives(f, cf.cf, cf.recourse, t)
                for o_name, o_val in obj_vals.items():
                    if o_name not in list(df_values.keys()):  # if not already evaluated
                        df_values[o_name] = o_val

            df_values['recourse'] = recourse
            df_values['cf_readable'] = self.env.writable_state(cf.cf)
            df_values['cf'] = [list(cf.cf)]

        df_values['fact'] = [list(f)]
        df_values['fact_readable'] = self.env.writable_state(f)
        df_values['target'] = t
        df_values['fact_id'] = fact_id

        header = not exists(self.eval_path)
        if not header:
            old_df = pd.read_csv(self.eval_path, header=0)
            df = pd.concat([old_df, pd.DataFrame(df_values,index=[0])], ignore_index=True)
            df.to_csv(self.eval_path, mode='w', index=None)
        else:
            df = pd.DataFrame(df_values, index=[0])
            df.to_csv(self.eval_path, mode='a', header=header, index=None)

    def generate_recourse(self, fact, cf, env, max_actions):
        states = []

        states.append((fact, []))
        level = 0
        expand = 10
        done = False

        while level <= max_actions and len(states) > 0 and not done:
            curr_state, curr_actions = states[-1]
            states = states[:-1]

            env.reset()
            env.set_state(curr_state)
            available_actions = env.get_actions(curr_state)

            for a in available_actions:
                for e in range(expand):
                    env.set_state(curr_state)
                    new_state, rew, done, _ = env.step(a)

                    states.append((new_state, curr_actions + [a]))

                    if env.equal_states(new_state, cf):
                        return curr_actions + [a]

            level += 1

        return 0

