import json
import math

import chess
import chess.engine
import torch
import pandas as pd

from src.envs.chessenv import ChessEnv
from src.models.chess_bb_model import ChessBBModel
from src.models.gridworld_bb_model import GridworldBBModel
from src.optimization.autoenc import AutoEncoder
from src.optimization.genetic_baseline import GeneticBaseline
from src.models.dataset import Dataset
from src.envs.gridworld import Gridworld
from src.objectives.baseline_objs import BaselineObjectives
from src.objectives.rl_objs import RLObjs
from src.tasks.task import Task
from src.optimization.monte_carlo_cfsearch import MCTSSearch
from src.utils.utils import seed_everything, load_fact


def main():
    seed_everything(seed=1)

    task_name = 'chess'

    # define paths
    model_path = 'trained_models/{}'.format(task_name)
    dataset_path = 'datasets/{}/dataset.csv'.format(task_name)
    fact_file = 'fact/{}.json'.format(task_name)
    param_file = 'params/{}.json'.format(task_name)

    if task_name == 'gridworld':
        env = Gridworld()
        bb_model = GridworldBBModel(env, model_path)
        enc_layers = [env.state_dim, 128, 16]
        max_actions = 5
    elif task_name == 'chess':
        engine_path = 'trained_models/stockfish_15.exe'
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        env = ChessEnv(engine)
        enc_layers = [env.state_dim, 512, 512, 32]
        max_actions = 1

        bb_model = ChessBBModel(env, engine_path)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {} Parameters = {}'.format(task_name, params))

    # define models
    dataset = Dataset(env, bb_model, dataset_path)
    train_dataset, test_dataset = dataset.split_dataset(frac=0.8)
    vae = AutoEncoder(layers=enc_layers)
    vae.fit(train_dataset, test_dataset)
    enc_data = vae.encode(torch.tensor(dataset._dataset.values))[0]

    # # define objectives
    baseline_obj = BaselineObjectives(env, bb_model, vae, enc_data, env.state_dim)
    rl_obj = RLObjs(env, bb_model, params, max_actions=max_actions)

    # get facts
    if task_name == 'gridworld':
        try:
            dataset_path = 'datasets/{}/facts.csv'.format(task_name)
            facts = pd.read_csv(dataset_path).values
        except FileNotFoundError:
            n_facts = 100
            facts = dataset._dataset.sample(n=n_facts)
            facts.to_csv(dataset_path, index=False)
        targets = None
    elif task_name == 'chess':
        facts, targets = load_fact(fact_file)

    # define methods n
    BO_GEN = GeneticBaseline(env, bb_model, dataset._dataset, baseline_obj, params)
    BO_MCTS = MCTSSearch(env, bb_model, dataset._dataset, baseline_obj, params, c=1)
    RL_MCTS = MCTSSearch(env, bb_model, dataset._dataset, rl_obj, params, c=1/math.sqrt(2))

    methods = [BO_GEN, BO_MCTS, RL_MCTS]
    method_names = ['BO_GEN', 'BO_MCTS', 'RL_MCTS']

    for i, m in enumerate(methods):
        print('\n------------------------ {} ---------------------------------------\n'.format(method_names[i]))
        eval_path = 'eval/{}/{}/rl_obj_results'.format(task_name, method_names[i])
        gen_nbhd = True if method_names[i] == 'BO_GEN' else False
        task = Task(task_name, env, bb_model, dataset, m, method_names[i], rl_obj, [rl_obj, baseline_obj], eval_path,
                    nbhd=gen_nbhd)

        faulty_list = [11, 13, 15, 17, 22, 25, 26, 27, 28, 38, 43, 44]

        task.run_experiment([facts[i] for i in range(len(facts)) if i in faulty_list], [targets[i] for i in range(len(targets)) if i in faulty_list])


if __name__ == '__main__':
    main()
