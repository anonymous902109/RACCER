import numpy as np
from sklearn import preprocessing as pre

class HighlightDiv:

    def __init__(self, env, bb_model, num_states=10):
        self.env = env
        self.bb_model = bb_model
        self.num_states = num_states

    def generate_important_states(self):
        summary_importances = []
        summary_states = []
        summary_actions = []

        cummulative_reward = 0
        cummulative_steps = 0

        num_simulations = 100
        runs = 0

        while runs < num_simulations:
            obs = self.env.reset()
            done = False
            steps = 0

            while not done:
                action = self.bb_model.predict(obs)
                Q_vals = self.bb_model.get_Q_vals(obs)

                # compute importance
                importance = max(Q_vals) - min(Q_vals)

                # check if frame should be added to summary
                if (len(summary_states) < self.num_states or importance > min(summary_importances)):
                    add = False
                    if len(summary_states) < self.num_states:
                        add = True
                    else:
                        most_similar_frame_idx = self.most_similar_state(obs, summary_states)

                        if summary_importances[most_similar_frame_idx] < importance:
                            # remove less important similar frame from summary
                            del summary_states[most_similar_frame_idx]
                            del summary_importances[most_similar_frame_idx]
                            del summary_actions[most_similar_frame_idx]
                            add = True

                    if add:
                        # add frame to summary
                        summary_states.append(obs)
                        summary_importances.append(importance)
                        summary_actions.append(action)

                steps += 1
                cummulative_steps += 1

                obs, reward, done, info = self.env.step(action)
                cummulative_reward += reward

                if done:
                    break

            runs += 1

        self.save_summary(summary_states, summary_importances)
        return summary_states

    def most_similar_state(self, state, added_states):
        differences = []
        q_vals_diffs = []
        state_diffs = []

        for s in added_states:
            q_vals_diff = sum(abs(np.subtract(np.array(self.bb_model.get_Q_vals(s)), np.array(self.bb_model.get_Q_vals(state)))))
            state_diff = sum(s!=state)
            state_diffs.append([state_diff])
            q_vals_diffs.append([q_vals_diff])

        state_diffs = pre.MinMaxScaler().fit_transform(np.array(state_diffs))
        q_vals_diffs = pre.MinMaxScaler().fit_transform(np.array(q_vals_diffs))

        state_diffs = list(state_diffs.squeeze())
        q_vals_diffs = list(q_vals_diffs.squeeze())

        differences = [q_vals_diffs[i] for i in range(len(q_vals_diffs))]

        min_diff_index = np.argmin(differences)

        return min_diff_index

    def save_summary(self, summary_states, summary_importances):
        for i, state in enumerate(summary_states):
            print('{} Importance = {}'.format(self.env.writable_state(state), summary_importances[i]))

    def select_important_states(self, states, indices):
        summary_importances = []
        summary_states = []
        summary_actions = []
        summary_indices = []

        for i, obs in enumerate(states):
            action = self.bb_model.predict(obs)
            Q_vals = self.bb_model.get_Q_vals(obs)

            # compute importance
            importance = max(Q_vals) - min(Q_vals)

            # check if frame should be added to summary
            if (len(summary_states) < self.num_states or importance > min(summary_importances)):
                add = False
                if len(summary_states) < self.num_states:
                    add = True
                else:
                    most_similar_frame_idx = self.most_similar_state(obs, summary_states)

                    if summary_importances[most_similar_frame_idx] < importance:
                        # remove less important similar frame from summary
                        del summary_states[most_similar_frame_idx]
                        del summary_indices[most_similar_frame_idx]
                        del summary_importances[most_similar_frame_idx]
                        del summary_actions[most_similar_frame_idx]
                        add = True

                if add:
                    # add frame to summary
                    summary_states.append(obs)
                    summary_indices.append(indices[i])
                    summary_importances.append(importance)
                    summary_actions.append(action)

        self.save_summary(summary_states, summary_importances)
        return summary_indices