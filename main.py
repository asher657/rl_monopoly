from statistical_tests import *
from utils import monopoly_logger
from agents.agent import Agent
from environment.board import Board
from agents.baseline_agent import BaselineAgent

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from datetime import datetime
import torch.multiprocessing as mp
import torch

from agents.dqnagent import DqnAgent
from utils.constants import DEFAULT_COST

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_agent(agent_type, logging_level, **kwargs):
    assert agent_type in ['random', 'baseline', 'dqn']

    if agent_type == 'random':
        return Agent(logging_level=logging_level)
    elif agent_type == 'baseline':
        return BaselineAgent(logging_level=logging_level)
    elif agent_type == 'dqn':
        if 'trained_policy_net' in kwargs and kwargs['trained_policy_net']:
            trained_policy_net_weights = torch.load(kwargs['trained_policy_net'], weights_only=True)
            trained_model_layer_sizes = []
            for key, tensor in trained_policy_net_weights.items():
                if 'weight' in key:
                    if 'output_layer' in key:
                        continue
                    trained_model_layer_sizes.append(tensor.shape[0])

            agent = DqnAgent(logging_level=logging_level, hidden_layer_sizes=trained_model_layer_sizes)
            agent.policy_net.load_state_dict(trained_policy_net_weights)
            agent.policy_net.eval()
            agent.eps = 0
        else:
            eps_decay = kwargs['num_episodes'] - (kwargs['num_episodes'] // 10)
            agent = DqnAgent(
                logging_level=logging_level,
                batch_size=kwargs.get('batch_size', 512),
                eps_decay=eps_decay,
                max_experience_len=kwargs.get('max_experience_len', 16384),
                lr=kwargs.get('lr', 0.001),
                update_target_net_freq=kwargs.get('update_target_net_freq', 5),
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', [256, 256, 256])
            )
        return agent


def run_episode(agent, episode, logging_level, default_cost, is_train=True):
    agent.reset()
    game_end = False

    board = Board(default_cost=default_cost, logging_level=logging_level)
    step = 0
    game_rewards = []

    while not game_end:
        curr_state = board.state
        next_move = agent.get_action(curr_state)
        reward, next_state, game_end = board.execute_action(next_move, agent, step)

        if is_train:
            agent.update_epsilon(episode)
            if agent.experience is not None:
                agent.experience.push((curr_state, next_move, reward, next_state, game_end))
            agent.optimize(episode)

        game_rewards.append(reward)
        step += 1

    agent_won = board.opponent_monies[-1] <= 0
    bought_positions = dict(Counter(board.bought_positions))
    bought_positions = {i: bought_positions.get(i, 0) for i in range(40)}

    return game_rewards, agent_won, bought_positions


def run_episodes(agent, num_episodes, logger, logging_level, default_cost, is_train=True):
    episode_rewards = []
    agent_wins = []
    bought_positions = {}
    episode_length = []
    for episode in range(num_episodes):
        logger.info(f'===== Starting Episode {episode} =====')
        rewards, won, episode_bought_positions = run_episode(agent, episode, logging_level, default_cost, is_train)
        episode_rewards.append(rewards)
        agent_wins.append(won)
        bought_positions[episode] = episode_bought_positions
        episode_length.append(len(rewards))
        logger.info('===== Agent won! =====' if won else '===== Opponent won! =====')
        logger.info(f'===== Game ended in {len(rewards)} steps =====')

    return episode_rewards, agent_wins, bought_positions, episode_length


def train(agent_type='baseline',
          num_episodes=100,
          logging_level='info',
          update_target_net_freq=5,
          batch_size=512,
          default_cost=500,
          max_experience_len=16384,
          lr=0.001,
          hidden_layer_sizes=[256, 256, 256],
          show_plots=True,
          save_model=True,
          run_date_time=None):
    logger = monopoly_logger.get_monopoly_logger(__name__, logging_level)
    logger.info(f'Starting Monopoly Game with {agent_type} agent!')

    agent = get_agent(agent_type,
                      logging_level,
                      num_episodes=num_episodes,
                      batch_size=batch_size,
                      update_target_net_freq=update_target_net_freq,
                      max_experience_len=max_experience_len,
                      lr=lr,
                      hidden_layer_sizes=hidden_layer_sizes)
    if agent_type == "dqn":
        agent.policy_net.train()
    episode_rewards, agent_wins, bought_positions, episode_length = run_episodes(agent, num_episodes, logger,
                                                                                 logging_level, default_cost)

    avg_rewards = [np.mean(x) for x in episode_rewards]
    if show_plots:
        plt.scatter(range(len(avg_rewards)), avg_rewards)
        plt.title(f'{agent_type} Average Episode Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig(f'plots/{agent_type}_training_rewards_{run_date_time}.png', dpi=300, bbox_inches='tight')
        plt.show()
        wins = [1 if i else 0 for i in agent_wins]
        win_rate = []
        cumulative_wins = 0
        for i in range(len(wins)):
            cumulative_wins += wins[i]
            win_rate.append(cumulative_wins / (i + 1))
        plt.plot(list(range(len(wins))), win_rate, marker='o')
        plt.xlabel("Games Played")
        plt.ylabel("Win Rate")
        plt.title("Win Rate Over Time")
        plt.savefig(f'plots/{agent_type}_latest_wins_{run_date_time}')
        plt.show()
        plt.scatter(range(len(episode_length)), episode_length)
        plt.title(f'{agent_type} Episode Length')
        plt.xlabel('Episodes')
        plt.ylabel('Length')
        plt.savefig(f'plots/{agent_type}_episode_length_{run_date_time}.png', dpi=300, bbox_inches='tight')
        plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(avg_rewards)}')
    logger.info(f'Agent last 50 average win rate: {np.mean(agent_wins[-50:])}')
    logger.info(f'Agent last 50 average reward: {np.mean(avg_rewards[-50:])}')

    if save_model:
        model_name = f'dqn_agent_{run_date_time}'
        agent.save_model(f'trained_agents/{model_name}')

    return avg_rewards, agent_wins


def evaluate(agent_type='dqn',
             num_episodes=100,
             logging_level='info',
             default_cost=500,
             trained_policy_net=None,
             run_date_time=None):
    logger = monopoly_logger.get_monopoly_logger(__name__, logging_level)
    logger.info(f'Starting Monopoly Game with {agent_type} agent!')

    agent = get_agent(agent_type, logging_level,
                      trained_policy_net=trained_policy_net,
                      num_episodes=num_episodes)
    if agent_type == "dqn":
        agent.policy_net.eval()
    episode_rewards, agent_wins, bought_positions, episode_length = run_episodes(agent, num_episodes, logger,
                                                                                 logging_level, default_cost,
                                                                                 is_train=False)

    avg_rewards = [np.mean(x) for x in episode_rewards]
    plt.scatter(range(len(avg_rewards)), avg_rewards)
    plt.title(f'{agent_type} Average Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig(f'plots/{agent_type}_evaluation_rewards_{run_date_time}.png', dpi=300, bbox_inches='tight')
    plt.show()
    win_rate = []
    cumulative_wins = 0
    for i in range(len(agent_wins)):
        cumulative_wins += agent_wins[i]
        win_rate.append(cumulative_wins / (i + 1))
    plt.plot(range(len(agent_wins)), win_rate, marker='o')
    plt.xlabel("Games Played")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Over Time")
    plt.savefig(f'plots/{agent_type}_latest_wins_{run_date_time}')
    plt.show()
    plt.scatter(range(len(episode_length)), episode_length)
    plt.title(f'{agent_type} Episode Length')
    plt.xlabel('Episodes')
    plt.ylabel('Length')
    plt.savefig(f'plots/{agent_type}_episode_length_{run_date_time}.png', dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(avg_rewards)}')

    return avg_rewards, agent_wins


def parallel_agent_training(args):
    (agent_type, num_episodes, logging_level, update_target_net_freq, batch_size,
     default_cost, max_experience_len, lr, hidden_layer_sizes, show_plots, save_model, run_date_time) = args
    return train(agent_type,
                 num_episodes,
                 logging_level,
                 update_target_net_freq,
                 batch_size,
                 default_cost,
                 max_experience_len,
                 lr,
                 hidden_layer_sizes,
                 show_plots,
                 save_model,
                 run_date_time)


def get_learning_curves(num_episodes: int = 30000,
                        logging_level='info',
                        update_target_net_freq=50,
                        batch_size=512,
                        default_cost=500,
                        max_experience_len=16384,
                        lr=0.001,
                        hidden_layer_sizes=[256],
                        run_date_time=None):
    agent_episode_rewards, agent_episode_wins = train("dqn",
                                                      num_episodes=num_episodes,
                                                      logging_level=logging_level,
                                                      update_target_net_freq=update_target_net_freq,
                                                      batch_size=batch_size,
                                                      default_cost=default_cost,
                                                      hidden_layer_sizes=hidden_layer_sizes,
                                                      max_experience_len=max_experience_len,
                                                      lr=lr,
                                                      save_model=False,
                                                      show_plots=False,
                                                      run_date_time=run_date_time)

    agent_episode_rewards = np.array(list(agent_episode_rewards))
    agent_episode_wins = np.array(list(agent_episode_wins))
    avg_agent_episode_wins = np.mean(agent_episode_wins, axis=0)
    right = 0
    sliding_sum_win = 0
    sliding_avg_win = []
    sliding_sum_reward = 0
    sliding_avg_reward = []
    for left in range(len(agent_episode_wins) - 199):
        while right < left + 200:
            sliding_sum_win += agent_episode_wins[right]
            sliding_sum_reward += agent_episode_rewards[right]
            right += 1
        sliding_avg_win.append(sliding_sum_win / 200)
        sliding_avg_reward.append(sliding_sum_reward / 200)
        sliding_sum_win -= agent_episode_wins[left]
        sliding_sum_reward -= agent_episode_rewards[left]

    fig, ax1 = plt.subplots()
    p1 = ax1.scatter(range(num_episodes), agent_episode_rewards, label="Average Reward of Episode", color="tab:blue")
    plt.title('DQN Learning Curve')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward Per Episode')
    p2 = ax1.plot(range(200, len(sliding_avg_reward) + 200), sliding_avg_reward, label="Sliding Reward", color="m")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Sliding Average Winrate")
    p3 = ax2.plot(range(200, len(sliding_avg_win) + 200), sliding_avg_win, label="Sliding Winrate", color="tab:red")
    plt.savefig(f'plots/dqn_learning_curve_{run_date_time}.png', dpi=300, bbox_inches='tight')
    fig.legend()
    plt.show()

    avg_agent_episode_rewards = np.mean(agent_episode_rewards, axis=0)
    return avg_agent_episode_rewards


def test_models(num_episodes: int = 10000,
                logging_level='info',
                update_target_net_freq=50,
                batch_size=512,
                default_cost=2000,
                max_experience_len=16384,
                lrs=[.001, .001, .001, .001, .0001, .0001, .0001, .0001],
                hidden_layer_sizes=[[256], [256, 256, 256], [256, 512, 256], [256, 512, 512, 256], [256],
                                    [256, 256, 256], [256, 512, 256], [256, 512, 512, 256]],
                run_date_time=datetime.now().strftime("%Y_%m_%d_%H_%M")):
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=min(len(lrs) + len(hidden_layer_sizes), mp.cpu_count())) as pool:
        args_list = [('dqn',
                      num_episodes,
                      logging_level,
                      update_target_net_freq,
                      batch_size,
                      default_cost,
                      max_experience_len,
                      lr,
                      hl,
                      False,
                      True,
                      run_date_time) for lr, hl in zip(lrs, hidden_layer_sizes)]
        results = pool.map(parallel_agent_training, args_list)

    agent_episode_rewards, agent_episode_wins = zip(*results)
    agent_episode_wins = np.array(list(agent_episode_wins))
    for i in range(len(agent_episode_wins)):
        agent_wins = list(agent_episode_wins[i, :])
        win_rate = []
        cumulative_wins = 0
        for j in range(len(agent_wins)):
            cumulative_wins += agent_wins[j]
            win_rate.append(cumulative_wins / (j + 1))
        plt.plot(range(len(agent_wins)), win_rate, marker='o')
        plt.xlabel("Games Played")
        plt.ylabel("Win Rate")
        plt.title("Win Rate Over Time")
        plt.show()
    return np.mean(agent_episode_wins, axis=1)


def run_statistical_tests(num_episodes, run_date_time, trained_policy_net='trained_agents/dqn_agent_final'):
    random_rewards, random_wins = evaluate(agent_type='random',
                                           num_episodes=num_episodes,
                                           logging_level='info',
                                           default_cost=DEFAULT_COST,
                                           trained_policy_net=trained_policy_net,
                                           run_date_time=run_date_time)
    baseline_rewards, baseline_wins = evaluate(agent_type='baseline',
                                               num_episodes=num_episodes,
                                               logging_level='info',
                                               default_cost=DEFAULT_COST,
                                               trained_policy_net=trained_policy_net,
                                               run_date_time=run_date_time)
    dqn_rewards, dqn_wins = evaluate(agent_type='dqn',
                                     num_episodes=num_episodes,
                                     logging_level='info',
                                     default_cost=DEFAULT_COST,
                                     trained_policy_net=trained_policy_net,
                                     run_date_time=run_date_time)

    run_analysis(random_wins, baseline_wins, dqn_wins, random_rewards, baseline_rewards, dqn_rewards, run_date_time)


if __name__ == '__main__':
    run_date_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    print(run_date_time)
    # train(agent_type='dqn',
    #       num_episodes=10000,
    #       logging_level='info',
    #       update_target_net_freq=50,
    #       batch_size=512,
    #       default_cost=DEFAULT_COST,
    #       max_experience_len=16384,
    #       lr=.001,
    #       hidden_layer_sizes=[256],
    #       run_date_time=run_date_time)
    # wr = test_models()
    # print(wr)

    # rewards, wins = evaluate(agent_type='dqn',
    #                          num_episodes=1000,
    #                          logging_level='info',
    #                          default_cost=DEFAULT_COST,
    #                          trained_policy_net='trained_agents/dqn_agent_final',
    #                          run_date_time=run_date_time)
    #
    # plt.hist(rewards)
    # plt.show()

    # get_learning_curves(num_episodes=20000, default_cost=DEFAULT_COST, hidden_layer_sizes=[256], run_date_time=run_date_time)

    trained_policy_net = 'trained_agents/dqn_agent_final'
    run_statistical_tests(1000, run_date_time, trained_policy_net)
