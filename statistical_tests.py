import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def perform_proportion_z_test(population, samples):
    population = np.array(population)
    samples = np.array(samples)
    population_success = np.unique(population, return_counts=True)[1][1]
    samples_success = np.unique(samples, return_counts=True)[1][1]

    successes = np.array([population_success, samples_success])
    n_obs = np.array([len(population), len(samples)])

    return proportions_ztest(successes, n_obs)


def perform_z_test(population, samples):
    population_mean = np.mean(population)
    samples_mean = np.mean(samples)

    population_std = np.std(population, ddof=0)
    n = len(samples)

    z = (samples_mean - population_mean) / (population_std / np.sqrt(n))

    p_value = 1 - stats.norm.cdf(abs(z))

    return z, p_value


def plot_wins(random_wins, baseline_wins, dqn_wins, run_date_time):
    df = pd.DataFrame({'random': random_wins, 'baseline': baseline_wins, 'dqn': dqn_wins})
    melted_df = pd.melt(df, var_name='agent', value_name='win')
    plt.title('Wins and Losses by Agent')
    sns.countplot(x='win', data=melted_df, hue='agent')
    plt.xlabel('Wins')
    plt.savefig(f'plots/agent_wins_losses_{run_date_time}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_rewards(random_rewards, baseline_rewards, dqn_rewards, run_date_time):
    plt.title('Rewards by Agent')
    plt.plot(random_rewards, label='random')
    plt.plot(baseline_rewards, label='baseline')
    plt.plot(dqn_rewards, label='dqn')
    plt.xlabel('Rewards')
    plt.legend()
    plt.savefig(f'plots/agent_rewards_{run_date_time}.png', dpi=300, bbox_inches='tight')
    plt.show()


def percent_change(x, y):
    return (y - x) / abs(x) * 100


def run_analysis(random_wins, baseline_wins, dqn_wins, random_rewards, baseline_rewards, dqn_rewards, run_date_time):
    random_win_z_score, random_win_p_value = perform_proportion_z_test(random_wins, dqn_wins)
    baseline_win_z_score, baseline_win_p_value = perform_proportion_z_test(baseline_wins, dqn_wins)
    random_reward_z_score, random_reward_p_value = perform_z_test(random_rewards, dqn_rewards)
    baseline_reward_z_score, baseline_reward_p_value = perform_z_test(baseline_rewards, dqn_rewards)
    plot_wins(random_wins, baseline_wins, dqn_wins, run_date_time)
    plot_rewards(random_rewards, baseline_rewards, dqn_rewards, run_date_time)

    random_avg_wins = np.mean(random_wins)
    baseline_avg_wins = np.mean(baseline_wins)
    dqn_avg_wins = np.mean(dqn_wins)
    random_avg_rewards = np.mean(random_rewards)
    baseline_avg_rewards = np.mean(baseline_rewards)
    dqn_avg_rewards = np.mean(dqn_rewards)

    with open('Final Results.txt', 'w') as f:
        print(f'#### AGENT WIN RATE ####', file=f)
        print(f'### Random vs DQN ###', file=f)
        print(f'Random Average Wins: {random_avg_wins:.2f}', file=f)
        print(f'DQN Average Wins: {dqn_avg_wins:.2f}', file=f)
        print(f'Win Rate Lift: {percent_change(random_avg_wins, dqn_avg_wins):.2f}%', file=f)
        print(f'Z-Score: {random_win_z_score:.5f}', file=f)
        print(f'p-value: {random_win_p_value:.5f}', file=f)
        print(f'Null Hypothesis: DQN Agent win rate is similar to the Random Agent', file=f)
        print(f'Alternate Hypothesis: DQN Agent win rate is different from the Random Agent', file=f)
        if random_win_p_value < 0.5:
            print(f'{random_win_p_value:.5f} < 0.5, therefore reject the null hypothesis', file=f)
        else:
            print(f'{random_win_p_value:.5f} >= 0.5, therefore fail to reject the null hypothesis', file=f)

        print('\n', file=f)
        print(f'### Baseline vs DQN ###', file=f)
        print(f'Baseline Average Wins: {baseline_avg_wins:.2f}', file=f)
        print(f'DQN Average Wins: {dqn_avg_wins:.2f}', file=f)
        print(f'Win Rate Lift: {percent_change(baseline_avg_wins, dqn_avg_wins):.2f}%', file=f)
        print(f'Z-Score: {baseline_win_z_score:.5f}', file=f)
        print(f'p-value: {baseline_win_p_value:.5f}', file=f)
        print(f'Null Hypothesis: DQN Agent win rate is similar to the Baseline Agent', file=f)
        print(f'Alternate Hypothesis: DQN Agent win rate is different from the Baseline Agent', file=f)
        if baseline_win_p_value < 0.5:
            print(f'{baseline_win_p_value:.5f} < 0.5, therefore reject the null hypothesis', file=f)
        else:
            print(f'{baseline_win_p_value:.5f} >= 0.5, therefore fail to reject the null hypothesis', file=f)

        print('\n', file=f)
        print(f'#### AGENT REWARDS ####', file=f)
        print(f'### Random vs DQN ###', file=f)
        print(f'Random Average Reward: {random_avg_rewards:.2f}', file=f)
        print(f'DQN Average Reward: {dqn_avg_rewards:.2f}', file=f)
        print(f'Reward Lift: {percent_change(random_avg_rewards, dqn_avg_rewards):.2f}%', file=f)
        print(f'Z-Score: {random_reward_z_score:.5f}', file=f)
        print(f'p-value: {random_reward_p_value:.5f}', file=f)
        print(f'Null Hypothesis: DQN Agent rewards are similar to the Random Agent', file=f)
        print(f'Alternate Hypothesis: DQN Agent rewards are different from the Random Agent', file=f)
        if random_reward_p_value < 0.5:
            print(f'{random_reward_p_value:.5f} < 0.5, therefore reject the null hypothesis', file=f)
        else:
            print(f'{random_reward_p_value:.5f} >= 0.5, therefore fail to reject the null hypothesis', file=f)

        print('\n', file=f)
        print(f'### Baseline vs DQN ###', file=f)
        print(f'Baseline Average Reward: {baseline_avg_rewards:.2f}', file=f)
        print(f'DQN Average Reward: {dqn_avg_rewards:.2f}', file=f)
        print(f'Reward Lift: {percent_change(baseline_avg_rewards, dqn_avg_rewards):.2f}%', file=f)
        print(f'Z-Score: {baseline_reward_z_score:.5f}', file=f)
        print(f'p-value: {baseline_reward_p_value:.5f}', file=f)
        print(f'Null Hypothesis: DQN Agent rewards are similar to the Baseline Agent', file=f)
        print(f'Alternate Hypothesis: DQN Agent rewards are different from the Baseline Agent', file=f)
        if baseline_reward_p_value < 0.5:
            print(f'{baseline_reward_p_value:.5f} < 0.5, therefore reject the null hypothesis', file=f)
        else:
            print(f'{baseline_reward_p_value:.5f} >= 0.5, therefore fail to reject the null hypothesis', file=f)
