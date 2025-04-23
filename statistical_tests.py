import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from statsmodels.stats.proportion import proportions_ztest


def perform_z_test(population, samples):
    population = np.array(population)
    samples = np.array(samples)
    population_success = np.unique(population, return_counts=True)[1][1]
    samples_success = np.unique(samples, return_counts=True)[1][1]

    successes = np.array([population_success, samples_success])
    n_obs = np.array([len(population), len(samples)])

    return proportions_ztest(successes, n_obs)


def plot_wins(random_wins, baseline_wins, dqn_wins, run_date_time):
    df = pd.DataFrame({'random': random_wins, 'baseline': baseline_wins, 'dqn': dqn_wins})
    melted_df = pd.melt(df, var_name='agent', value_name='win')
    plt.title('Wins and Losses by Agent')
    sns.countplot(x='win', data=melted_df, hue='agent')
    plt.xlabel('Wins')
    plt.savefig(f'plots/agent_wins_losses_{run_date_time}.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_analysis(random_wins, baseline_wins, dqn_wins, run_date_time):
    random_z_score, random_p_value = perform_z_test(random_wins, dqn_wins)
    baseline_z_score, baseline_p_value = perform_z_test(baseline_wins, dqn_wins)
    plot_wins(random_wins, baseline_wins, dqn_wins, run_date_time)
    with open('Final Results.txt', 'w') as f:
        print(f'### Random vs DQN ###', file=f)
        print(f'Z-Score: {random_z_score:.5f}', file=f)
        print(f'p-value: {random_p_value:.5f}', file=f)
        print(f'Null Hypothesis: DQN Agent performs worse or the same as the Random Agent', file=f)
        print(f'Alternate Hypothesis: DQN Agent performs better than the Random Agent', file=f)
        if random_p_value < 0.5:
            print(f'{random_p_value:.5f} < 0.5, therefore reject the null hypothesis', file=f)
        else:
            print(f'{random_p_value:.5f} >= 0.5, therefore fail to reject the null hypothesis', file=f)

        print('\n', file=f)
        print(f'### Baseline vs DQN ###', file=f)
        print(f'Z-Score: {baseline_z_score:.5f}', file=f)
        print(f'p-value: {baseline_p_value:.5f}', file=f)
        print(f'Null Hypothesis: DQN Agent performs worse or the same as the Baseline Agent', file=f)
        print(f'Alternate Hypothesis: DQN Agent performs better than the Baseline Agent', file=f)
        if baseline_p_value < 0.5:
            print(f'{baseline_p_value:.5f} < 0.5, therefore reject the null hypothesis', file=f)
        else:
            print(f'{baseline_p_value:.5f} >= 0.5, therefore fail to reject the null hypothesis', file=f)
