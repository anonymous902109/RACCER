import random

import pandas as pd
import numpy as np


def evaluate_coverage(eval_path, method_name='', total_facts=100):
    df = pd.read_csv(eval_path, header=0)
    facts = pd.unique(df['fact_id'])

    coverage = (1.0*len(facts)) / total_facts
    print('Method = {}. Average number of cfs per fact for {} facts = {}'.format(method_name, len(facts), coverage))
    return coverage


def evaluate_quantity(eval_path, method_name=''):
    df = pd.read_csv(eval_path, header=0)
    facts = pd.unique(df['fact_id'])

    cfs = []
    for f in facts:
        n = len(df[df['fact_id'] == f])
        cfs.append(n)

    print('Method = {}. Average number of cfs per fact for {} facts = {}'.format(method_name, len(facts), np.mean(cfs)))
    return np.mean(cfs)

def evaluate_diversity(eval_path, method_name=''):
    df = pd.read_csv(eval_path, header=0)
    facts = pd.unique(df['fact_id'])

    metrics = ['cost', 'validity', 'reachability']

    diversity = []

    for f in facts:
        df_fact = df[df['fact_id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = 0
                    for m in metrics:
                        diff += (x[m] - y[m]) ** 2

                    diversity.append(diff)

    avg_div = np.mean(diversity)
    print('Method = {}. Average diversity for {} facts = {}'.format(method_name, len(facts), avg_div))

    return avg_div

def evaluate_all(tasks, agent_types, method_names, eval_objs):
    for t in tasks:
        for a in agent_types:
            for m in method_names:
                eval_path = 'eval/{}/{}/{}'.format(t, m, a)
                try:
                    df = pd.read_csv(eval_path, header=0)
                    print('Task = {} Agent_type = {}'.format(t, a))
                    evaluate_objectives(df, eval_objs, m)
                    print('----------------------------------')

                except FileNotFoundError:
                    continue

def evaluate_objectives(df, eval_objs, method_name=''):
    eval_obj_names = []
    df = df[df['cf'] != '0']

    for eo in eval_objs:
        eval_obj_names += eo.objectives

    res = {}
    for eo_name in eval_obj_names:
        if (eo_name == 'validity' or eo_name == 'realistic') and df[eo_name].dtype == object:
            df[eo_name] = df[eo_name].map({'True': True, 'False': False}).astype(bool)
        res[eo_name] = round(np.mean(df[eo_name].values), 2)

    print('Method = {}. Average values for objectives = {}'.format(method_name, res))

def get_realistic_df(eval_paths, targets=None, p=0.5):
    filtered_facts = []
    for ep in eval_paths:
        df = pd.read_csv(ep, header=0)

        if targets is not None:
            df = df[df.target.isin(targets)]

        grouped_df = df[df.target.isin(targets)][['fact_id', 'realistic']].groupby(['fact_id']).prod()
        facts = grouped_df[grouped_df['realistic'] != 0].index
        facts = list(facts)

        if len(filtered_facts) == 0:
            filtered_facts = facts

        filtered_facts = [f for f in filtered_facts if f in facts]

    filter_df = df[df['fact_id'].isin(filtered_facts)]

    return filter_df

def split_df(df, p=0.5):
    unique_facts = list(df.fact_id.unique())
    sample_facts = random.sample(unique_facts, int(p*len(unique_facts)))

    train_df = df[df.fact_id.isin(sample_facts)]
    test_df = df.drop(train_df.index)

    train_df_facts = list(train_df.fact_id.unique())
    test_df_facts = list(test_df.fact_id.unique())

    return (train_df_facts, test_df_facts)

def print_summary_split(df, train_df_facts, test_df_facts, eval_paths, targets):

    print('--------- USER STUDY TRAINING DATASET ------------')
    for fact_id in train_df_facts:
        print('FACT #{}:'.format(fact_id))
        fact_df = df[df['fact_id'] == fact_id]
        print('{}'.format(fact_df.sample(1)['fact_readable'].values))
        for e in eval_paths:
            df = pd.read_csv(e, header=0)
            method = e.split('/')[-2]
            cfs = df[(df['fact_id'] == fact_id) & (df['target'].isin(targets))]
            for _, cf in cfs.iterrows():
                print('\tCounterfactual for method = {} target = {}: {}'.format(method, cf['target'], cf['cf_readable']))
            print('\t------------------------------------')
        print('------------------------------------')

    print('--------- USER STUDY TEST DATASET ------------')
    for fact_id in test_df_facts:
        print('FACT #{}:'.format(fact_id))
        fact_df = df[df['fact_id'] == fact_id]
        print('{}'.format(fact_df.sample(1)['fact_readable'].values))
        for e in eval_paths:
            df = pd.read_csv(e, header=0)
            method = e.split('/')[-2]
            cfs = df[(df['fact_id'] == fact_id) & (df['target'].isin(targets))]
            for _, cf in cfs.iterrows():
                print(
                    '\tCounterfactual for method = {} target = {}: {}'.format(method, cf['target'], cf['cf_readable']))
            print('\t------------------------------------')






