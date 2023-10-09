import torch

from src.baselines.abs_baseline import AbstractBaseline
from src.baselines.models.enc_dec import EncoderDecoder
from src.baselines.models.generator import Generator

from src.models.counterfactual import CF

import numpy as np


class GANterfactual(AbstractBaseline):

    def __init__(self, env, bb_model, params, generator_path):
        self.env = env
        self.bb_model = bb_model
        self.params = params
        self.nb_domains = self.env.action_space.n

        self.generator = Generator(params["layer_shapes"])
        self.generator.load_state_dict(torch.load(generator_path))

        super(GANterfactual, self).__init__()

    def generate_counterfactuals(self, fact, target):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target):
        ''' Returns all cfs found in the tree '''
        tensor_fact = torch.tensor(fact, dtype=float).unsqueeze(0)
        cf = self.generate_counterfactual(tensor_fact, target, self.nb_domains)

        cf = cf.squeeze().tolist()
        cf = [round(feature) for feature in cf]

        cf = CF([], cf, {}, 0)
        return cf

    def generate_counterfactual(self, fact, target, nb_domains):
        # convert target class to onehot
        onehot_target_class = np.zeros(nb_domains, dtype=int)
        onehot_target_class[target] = 1
        onehot_target_class = torch.tensor([onehot_target_class])

        # generate counterfactual
        counterfactual = self.generator(fact, onehot_target_class)

        return counterfactual


