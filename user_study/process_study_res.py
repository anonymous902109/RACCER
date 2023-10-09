import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main():
    df_R = pd.read_csv('raccer.csv', header=0)
    df_G = pd.read_csv('gan.csv', header=0)

    # calculating accuracy
    correct_answers = ['Shoot', 'Down', 'right', 'down', 'right', 'right', 'right', 'shoot', 'shoot', 'shoot',
                       'shoot', 'shoot', 'right', 'down', 'down', 'shoot', 'shoot', 'right', 'right', 'shoot']

    correct_choices = ['Agent B', 'Agent A']

    correct_answers = [ca.upper() for ca in correct_answers]

    correct_R = []
    correct_choices_R = []
    for index, row in df_R.iterrows():
        answers = row[9:-17].dropna().values
        answers = [answers[i] for i in range(len(answers)) if i not in [5, 10, 19]]
        n = sum([answers[i] == correct_answers[i] for i in range(len(correct_answers))])
        correct_R.append(n)

        cc = (str(row[-16]) == correct_choices[0]) + (str(row[-14]) == correct_choices[1])
        correct_choices_R.append(cc)

    correct_G = []
    correct_choices_G = []
    for index, row in df_G.iterrows():
        answers = row[9:-17].dropna().values
        answers = [answers[i] for i in range(len(answers)) if i not in [5, 10, 19]]
        n = sum([answers[i] == correct_answers[i] for i in range(len(correct_answers))])
        correct_G.append(n)

        cc = (str(row[-16]) == correct_choices[0]) + (str(row[-14]) == correct_choices[1])
        correct_choices_G.append(cc)

    stat, p = mannwhitneyu(correct_G, correct_R, method="exact")

    print('Accuracy: Stat = {} p value = {}'.format(stat, p))

    stat, p = mannwhitneyu(correct_choices_G, correct_choices_R, method="exact")

    print('Agent choice: Stat = {} p value = {}'.format(stat, p))

    # calculating expl satisfaction
    expl_satisfaction_cols = df_R.columns[-11:-4]

    satisfaction_values_R = {}
    satisfaction_values_G = {}
    c_names = ['Useful', 'Satisfying', 'Detailed', 'Complete', 'Actionable', 'Reliable', 'Trustworthy']

    for i, c in enumerate(expl_satisfaction_cols):

        s_R = df_R[c].fillna(0).values.squeeze()
        s_G = df_G[c].fillna(0).values.squeeze()

        stat, p = mannwhitneyu(s_R, s_G, method="exact")

        print('Feature c = {}, stat = {} p value = {}'.format(c, stat, p))

        satisfaction_values_R[c_names[i]] = s_R
        satisfaction_values_G[c_names[i]] = s_G

    plot_df_R = pd.DataFrame.from_dict(satisfaction_values_R)
    plot_df_G = pd.DataFrame.from_dict(satisfaction_values_G)
    label = ['RACCER'] * len(plot_df_R) + ['GANterfactual-RL'] * len(plot_df_G)

    plot_df = pd.concat([plot_df_R, plot_df_G])
    plot_df['label'] = label

    plot_df = pd.melt(plot_df, id_vars=['label'], value_vars=c_names)

    ax = sns.barplot(data=plot_df, x="variable", y="value", hue="label")

    ax.set(xlabel='', ylabel='Score')
    ax.set_yticks(np.arange(0, 5.5, step=0.5))
    plt.ylim(0, 5)
    plt.legend(title='Algorithm')
    plt.show()


if __name__ == '__main__':
    main()