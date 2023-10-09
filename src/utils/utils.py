import json
import os
import random

import torch
import pandas as pd
import numpy as np

from src.utils.highligh_div import HighlightDiv


def seed_everything(seed):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # tf.random.set_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)


def load_facts_from_summary(env, bb_model, num_states=10):
    highlight = HighlightDiv(env, bb_model, num_states=30)
    facts = highlight.generate_important_states()

    return facts, [[4, 5]] * len(facts)

def generate_summary_states(env, bb_model, df):
    states = df['fact'].values
    trans_states = []
    ind = 0
    indices = []

    for s in states:

        s = s.strip('][').split(', ')
        s = [int(r) for r in s]
        if s not in trans_states:
            trans_states.append(s)
            indices.append(ind)
        ind += 1

    trans_states = np.array(trans_states)

    highlight = HighlightDiv(env, bb_model, num_states=20)
    summary_state_ind = highlight.select_important_states(trans_states, indices)

    df.reset_index(inplace=True)
    return df[df.index.isin(summary_state_ind)]

def load_facts_from_json(fact_file):
    with open(fact_file, 'r') as f:
        content = json.load(f)

    facts = []
    targets = []
    for fact in content:
        facts.append(fact)
        target = fact['target']
        targets.append([target])

    return facts, targets


def load_facts_from_csv(csv_path, env, bb_model, n_ep=100):
    try:
        df = pd.read_csv(csv_path, header=0)
        facts = df.values
        return facts, None

    except FileNotFoundError:
        print('Generating facts...')
        data = []

        for i in range(n_ep):
            obs = env.reset()
            done = False

            while not done:
                data += [list(obs)]

                choice = random.randint(0, 1)
                action = bb_model.predict(obs) if choice == 0 else env.action_space.sample()

                obs, rew, done, _ = env.step(action)

        dataframe = pd.DataFrame(data)
        dataframe.drop_duplicates(inplace=True)

        dataframe = dataframe.sample(100)

        dataframe.to_csv(csv_path, index=None)

        return dataframe.values, None