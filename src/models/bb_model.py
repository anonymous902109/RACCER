import torch
from stable_baselines3 import DQN


class BBModel:

    def __init__(self, env, model_path, training_timesteps):
        self.model_path = model_path
        self.env = env
        self.training_timesteps = training_timesteps

        self.model = self.load_model(model_path, env)

    def load_model(self, model_path, env):
        try:
            model = DQN.load(model_path)
            model.env = env
            print('Loaded bb model')
        except FileNotFoundError:
            print('Training bb model')
            model = DQN('MlpPolicy',
                        env,
                        verbose=1,
                        exploration_fraction=0.8,  # 0.8
                        policy_kwargs={'net_arch': [256, 256]},  # 256, 256
                        learning_rate=0.0001,
                        learning_starts=200,
                        batch_size=128,
                        gamma=0.98,
                        train_freq=1,
                        gradient_steps=1,
                        target_update_interval=50,
                        )
            model.learn(total_timesteps=self.training_timesteps)
            model.save(model_path)
        return model

    def predict(self, x):
        action, _ = self.model.predict(x, deterministic=True)

        return action.item()

    def get_action_prob(self, x, a):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)
        probs = torch.softmax(q_values, dim=-1).squeeze()

        return probs[a].item()

    def get_Q_vals(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)

        return q_values.squeeze().tolist()