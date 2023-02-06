import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():
    user_study_A_path = 'eval/user_study/study_A.csv'
    user_study_B_path = 'eval/user_study/study_B.csv'

    correct_answers_path = 'eval/user_study/answers.json'

    properties = ['Understand', 'Satisfaction', 'Detailed', 'Complete', 'Actionable', 'Reliability', 'Trustworthy', 'Confidence']

    study_names = ['BO+GEN', 'RACCER']
    prop_df = None
    paths = [user_study_A_path, user_study_B_path]
    dfs = []

    for i, p in enumerate(paths):
        try:
            df = pd.read_csv(p, header=0)
            df = df.dropna(axis=0, how='all')
            df = df.dropna(axis=1, how='all')
            with open(correct_answers_path, 'r') as f:
                answers = json.load(f)

            print('-------------- Loaded dataset at {} -------------- '.format(p))

            correct_per_user = []
            for j in range(len(df)):
                partic = df.iloc[j]
                correct_for_partic = 0
                for q in list(answers.keys()):
                    correct_for_partic += (partic[q] == answers[q])

                correct_per_user.append(correct_for_partic)

            print('Approach = {} Correct per user = {}'.format(study_names[i], correct_per_user))

            correct = 0.0
            participants = len(df)
            questions = len(answers)
            print('Participants = {} Questions = {}'.format(participants, questions))
            for q in list(answers.keys()):
                correct_q = sum(df[q] == answers[q])
                correct += correct_q
                print('Correct for {} = {}'.format(q, correct_q))

            print('Correct questions = {}. Percentage = {}'.format(correct, correct/(participants * questions)))

            # Calculate average ratings for individual properties
            prop_results = {}
            for prop in properties:
                rating = sum(df[prop]) / participants
                print('{} = {}'.format(prop, rating))
                prop_results[prop] = [rating]

            # Create dataset for individual studies
            prop_results['Model'] = study_names[i]
            if prop_df is None:
                prop_df = pd.DataFrame.from_dict(prop_results)
            else:
                prop_df = pd.concat([prop_df, pd.DataFrame.from_dict(prop_results)])

            dfs.append(df)
        except FileNotFoundError:
            print('File not found at {}'.format(p))

    for p in properties:
        t1 = dfs[0][p].values
        t2 = dfs[1][p].values

        d = t1 - t2

        from scipy.stats import wilcoxon
        res = wilcoxon(d)

        print('Statistical testing for {} = {}'.format(p, res))

    # Transform dataframe
    prop_df.rename(columns={'Understand': 'Understanding',
                            'Detailed': 'Detail',
                            'Complete': 'Completeness',
                            'Actionable': 'Actionability',
                            'Trustworthy': 'Trust'}, inplace=True)
    prop_df = pd.melt(prop_df, id_vars='Model', value_name='Rating', var_name='Property')

    # Plot results
    sns.barplot(data=prop_df, x="Property", y="Rating", hue="Model")
    plt.xlabel('')
    plt.yticks(ticks=np.arange(0, 5, step=0.5))
    # plt.xticks(ticks=[], labels=['Understanding', 'Satisfaction', 'Detailed', 'Completeness', 'Actionablity', 'Reliability', 'Trust', 'Confidence'])
    plt.xticks(rotation=45)
    plt.ylabel('Likert Rating')
    plt.show()



if __name__ == '__main__':
    main()