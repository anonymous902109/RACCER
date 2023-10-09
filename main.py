import json

from src.baselines.fid_raccer import FidRACCER
from src.baselines.ganterfactual import GANterfactual
from src.baselines.mcts_raccer import MCTSRACCER

from src.envs.farm0 import Farm0
from src.envs.frozen_lake import FrozenLake
from src.envs.gridworld import Gridworld
from src.evaluation.eval import evaluate_objectives, get_realistic_df, split_df, print_summary_split, evaluate_all
from src.models.bb_model import BBModel
from src.optimization.objs.fid_obj import FidObj
from src.optimization.objs.game_obj import GameObj
from src.optimization.objs.sl_obj import SLObj
from src.tasks.task import Task
from src.utils.utils import seed_everything, load_facts_from_summary, load_facts_from_csv, generate_summary_states


def main(task_name, agent_type):
    print('TASK = {} AGENT_TYPE = {}'.format(task_name, agent_type))
    seed_everything(seed=1)

    training_timesteps = 300000
    if agent_type == 'suboptim':
        training_timesteps = training_timesteps / 10

    # define paths
    model_path = 'trained_models/{}_{}'.format(task_name, agent_type)
    fact_csv_dataset_path = 'datasets/{}/facts.csv'.format(task_name, agent_type)
    fact_json_path = 'fact/{}.json'.format(task_name)
    param_file = 'params/{}.json'.format(task_name)
    generator_path = 'trained_models/generator_{}_{}.ckpt'.format(task_name, agent_type)

    if task_name == 'gridworld':
        env = Gridworld()
        gym_env = env
    elif task_name == 'frozen_lake':
        env = FrozenLake()
        gym_env = env
    elif task_name == 'farm0':
        env = Farm0()
        gym_env = env

    bb_model = BBModel(gym_env, model_path, training_timesteps)

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)
        print('Task = {}\nParameters = {}'.format(task_name, params))

    # get facts
    facts, targets = load_facts_from_csv(fact_csv_dataset_path, env, bb_model)
    # facts, targets = load_facts_from_json(fact_json_path)
    # facts, targets = load_facts_from_summary(env, bb_model)

    # define methods
    fid_raccer = FidRACCER(env, bb_model, params)
    mcts_raccer = MCTSRACCER(env, bb_model, params)
    ganterfactual = GANterfactual(env, bb_model, params, generator_path)

    methods = [fid_raccer, ganterfactual]
    method_names = ['FidRACCER', 'GANterfactual']

    # define objectives
    sl_obj = SLObj(env, bb_model, params)
    game_obj = GameObj(env, bb_model, params)
    fid_obj = FidObj(env, bb_model, params)

    # define eval objectives
    eval_objs = [[sl_obj, game_obj, fid_obj], [sl_obj, game_obj, fid_obj]]

    # for i, m in enumerate(methods):
    #     print('\n------------------------ {} ---------------------------------------\n'.format(method_names[i]))
    #     eval_path = 'eval/{}/{}/{}'.format(task_name, method_names[i], agent_type)
    #     task = Task(task_name, env, bb_model, m, method_names[i], eval_path, eval_objs[i], params)
    #
    #     task.run_experiment(facts[1:],  targets)
    #
    evaluate_all(tasks=['frozen_lake', 'gridworld', 'farm0'],
                 agent_types=['optim', 'suboptim', 'non_optim'],
                 method_names=method_names,
                 eval_objs=[sl_obj, game_obj, fid_obj])

    # eval_path_template = 'eval/{}/{}/{}'
    # eval_paths = [eval_path_template.format(task_name, method_name, agent_type) for method_name in method_names]
    # realistic_df = get_realistic_df(eval_paths, targets=[4, 5])
    # summary = generate_summary_states(env, bb_model, realistic_df)
    # indices = split_df(summary)
    # #
    # print_summary_split(summary, *indices, eval_paths, targets=[4, 5])


if __name__ == '__main__':
    # main('farm0', 'optim')
    # main('farm0', 'suboptim')

    # main('gridworld', 'optim')
    main('gridworld', 'non_optim')
    # main('gridworld', 'suboptim')
    #
    # main('frozen_lake', 'optim')
    # main('frozen_lake', 'suboptim')
