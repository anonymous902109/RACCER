from stable_baselines3 import DQN


class GridworldBBModel():

    def __init__(self, env, model_path):
        self.model_path = model_path
        self.env = env
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
                        exploration_fraction=0.8,
                        policy_kwargs={'net_arch': [256, 256]},
                        learning_rate=0.0001,
                        learning_starts=200,
                        batch_size=32,
                        gamma=0.98,
                        train_freq=1,
                        gradient_steps=1,
                        target_update_interval=50,
                        )
            model.learn(total_timesteps=500000)
            model.save(model_path)
        return model

    def predict(self, x):
        action, _ = self.model.predict(x, deterministic=True)

        return action