import numpy as np
import pandas as pd
from tqdm import tqdm


class Dataset:

    def __init__(self, env, bb_model, dataset_path):
        self.env = env
        self.bb_model = bb_model
        self.dataset_path = dataset_path

        self._dataset = self.generate_dataset(env, bb_model, dataset_path)

    def generate_dataset(self, env, model, dataset_path, n_ep=200):
        try:
            df = pd.read_csv(dataset_path, index_col=False)
            print('Loaded dataset with {} samples'.format(len(df)))
        except FileNotFoundError:
            print('Generating dataset...')
            ds = []

            for i in tqdm(range(n_ep)):
                obs = env.reset()
                done = False
                c = 0
                while (not done) and (c < 50):
                    c += 1
                    ds.append(list(obs))
                    rand = np.random.randint(0, 2)
                    if rand == 0:
                        action = model.predict(obs)
                    else:
                        action = np.random.choice(env.get_actions(obs))

                    obs, rew, done,  _ = env.step(action)

            df = pd.DataFrame(ds)
            df = df.drop_duplicates()

            print('Generated {} samples!'.format(len(df)))
            df.to_csv(dataset_path, index=False)

        return df

    def split_dataset(self, frac=0.8):
        train_dataset = self._dataset.sample(frac=0.8, random_state=1)
        test_dataset = self._dataset.drop(train_dataset.index)

        return train_dataset, test_dataset