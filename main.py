import monopoly_logger
from agent import Agent
from board import Board
from baseline_agent import BaselineAgent

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from datetime import datetime
import torch.multiprocessing as mp
import torch

from dqnagent import DqnAgent

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
    episode_rewards, agent_wins, bought_positions, episode_length = run_episodes(agent, num_episodes, logger, logging_level, default_cost)

    avg_rewards = [np.mean(x) for x in episode_rewards]
    if show_plots:
        plt.scatter(range(len(avg_rewards)), avg_rewards)
        plt.title(f'{agent_type} Average Episode Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig(f'plots/{agent_type}_training_rewards_{run_date_time}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        wins = [1 if i else 0 for i in agent_wins]
        win_rate = []
        cumulative_wins = 0
        for i in range(len(wins)):
            cumulative_wins += wins[i]
            win_rate.append(cumulative_wins / (i + 1))
        plt.plot(list(range(wins)), win_rate, marker='o')
        plt.xlabel("Games Played")
        plt.ylabel("Win Rate")
        plt.title("Win Rate Over Time")
        # plt.show()
        plt.savefig(f'plots/{agent_type}_latest_wins_{run_date_time}')
        plt.scatter(range(len(episode_length)), episode_length)
        plt.title(f'{agent_type} Episode Length')
        plt.xlabel('Episodes')
        plt.ylabel('Length')
        plt.savefig(f'plots/{agent_type}_episode_length_{run_date_time}.png', dpi=300, bbox_inches='tight')
        # plt.show()


    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(avg_rewards)}')
    logger.info(f'Agent last 50 average win rate: {np.mean(agent_wins[-50:])}')
    logger.info(f'Agent last 50 average reward: {np.mean(avg_rewards[-50:])}')

    if save_model:
        model_name = f'dqn_agent_{run_date_time}'
        agent.save_model(f'trained_agents/{model_name}')

    return episode_rewards


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
    episode_rewards, agent_wins, bought_positions, episode_length = run_episodes(agent, num_episodes, logger, logging_level, default_cost, is_train=False)

    avg_rewards = [np.mean(x) for x in episode_rewards]
    plt.scatter(range(len(avg_rewards)), avg_rewards)
    plt.title(f'{agent_type} Average Episode Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig(f'plots/{agent_type}_evaluation_rewards_{run_date_time}.png', dpi=300, bbox_inches='tight')
    plt.show()
    wins = [1 if i else 0 for i in agent_wins]
    win_rate = []
    cumulative_wins = 0
    for i in range(len(wins)):
        cumulative_wins += wins[i]
        win_rate.append(cumulative_wins / (i + 1))
    plt.plot(list(range(wins)), win_rate, marker='o')
    plt.xlabel("Games Played")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Over Time")
    plt.show()
    plt.savefig(f'plots/{agent_type}_latest_wins_{run_date_time}')
    plt.scatter(range(len(episode_length)), episode_length)
    plt.title(f'{agent_type} Episode Length')
    plt.xlabel('Episodes')
    plt.ylabel('Length')
    plt.savefig(f'plots/{agent_type}_episode_length_{run_date_time}.png', dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(avg_rewards)}')

    return avg_rewards


def parallel_agent_training(args):
    (agent_type, num_episodes, logging_level, update_target_net_freq, batch_size,
     default_cost, max_experience_len, lr, hidden_layer_sizes, show_plots, save_model) = args
    agent_rewards = train(agent_type,
                          num_episodes,
                          logging_level,
                          update_target_net_freq,
                          batch_size,
                          default_cost,
                          max_experience_len,
                          lr,
                          hidden_layer_sizes,
                          show_plots,
                          save_model)

    return [np.mean(x) for x in agent_rewards]


def get_learning_curves(num_agents: int = 50,
                        num_episodes: int = 30000,
                        logging_level='info',
                        update_target_net_freq=50,
                        batch_size=512,
                        default_cost=500,
                        max_experience_len=16384,
                        lr=0.001,
                        hidden_layer_sizes=[256],
                        run_date_time=None):
    mp.set_start_method('spawn', force=True)

    with mp.Pool(processes=min(num_agents, mp.cpu_count())) as pool:
        args_list = [('dqn',
                      num_episodes,
                      logging_level,
                      update_target_net_freq,
                      batch_size,
                      default_cost,
                      max_experience_len,
                      lr,
                      hidden_layer_sizes,
                      False,
                      False) for _ in range(num_agents)]

        agent_episode_rewards = pool.map(parallel_agent_training, args_list)

    agent_episode_rewards = np.array(agent_episode_rewards)
    avg_agent_episode_rewards = np.mean(agent_episode_rewards, axis=1)
    return np.argmax(avg_agent_episode_rewards)


def test_models(num_episodes: int = 30000,
                logging_level='info',
                update_target_net_freq=50,
                batch_size=512,
                default_cost=500,
                max_experience_len=16384,
                lrs=[.001,.001,.001,.001,.0001,.0001,.0001,.0001],
                hidden_layer_sizes=[[256],[256,256,256],[256,512,256],[256,512,512,256],[256],[256,256,256],[256,512,256],[256,512,512,256]],
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
                True,
                True,
                run_date_time) for lr, hl in zip(lrs, hidden_layer_sizes)]
        agent_episode_rewards = pool.map(parallel_agent_training, args_list)
    agent_episode_rewards = np.array(agent_episode_rewards)
    mean_rewards = agent_episode_rewards[:,-50:]
    return np.argmax(mean_rewards)

if __name__ == '__main__':
    run_date_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    DEFAULT_COST = 2000
    # lrs = [.001,.001,.001,.001,.0001,.0001,.0001,.0001]
    # hidden_sizes = [[256],[256,256,256],[256,512,256],[256,512,512,256],[256],[256,256,256],[256,512,256],[256,512,512,256]]
    # for lr, hs in zip(lrs, hidden_sizes):
    train(agent_type='dqn',
        num_episodes=10000,
        logging_level='info',
        update_target_net_freq=50,
        batch_size=512,
        default_cost=DEFAULT_COST,
        max_experience_len=16384,
        lr=.001,
        hidden_layer_sizes=[256],
        run_date_time=run_date_time)
    # trained_policy_net = 'trained_agents/dqn_agent_2025_04_17_22_42'
    # evaluate(agent_type='baseline',
    #          num_episodes=1000,
    #          logging_level='info',
    #          default_cost=DEFAULT_COST,
    #          trained_policy_net=trained_policy_net,
    #          run_date_time=run_date_time)
    # get_learning_curves(5, num_episodes=1000, run_date_time=run_date_time)
