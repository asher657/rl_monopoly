import monopoly_logger
from agent import Agent
from board import Board
from baseline_agent import BaselineAgent

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
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
    return game_rewards, agent_won


def run_episodes(agent, num_episodes, logger, logging_level, default_cost, is_train=True):
    episode_rewards = []
    agent_wins = []

    for episode in range(num_episodes):
        logger.info(f'===== Starting Episode {episode} =====')
        rewards, won = run_episode(agent, episode, logging_level, default_cost, is_train)
        episode_rewards.append(rewards)
        agent_wins.append(won)

        logger.info('===== Agent won! =====' if won else '===== Opponent won! =====')
        logger.info(f'===== Game ended in {len(rewards)} steps =====')

    return episode_rewards, agent_wins


def train(agent_type='baseline',
          num_episodes=100,
          logging_level='info',
          update_target_net_freq=5,
          batch_size=512,
          default_cost=500,
          max_experience_len=16384,
          lr=0.001,
          hidden_layer_sizes=[256, 256, 256]):
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

    episode_rewards, agent_wins = run_episodes(agent, num_episodes, logger, logging_level, default_cost)

    avg_rewards = [np.mean(x) for x in episode_rewards]
    plt.scatter(range(len(avg_rewards)), avg_rewards)
    plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(avg_rewards)}')
    logger.info(f'Agent last 50 average win rate: {np.mean(agent_wins[-50:])}')
    logger.info(f'Agent last 50 average reward: {np.mean(avg_rewards[-50:])}')

    model_name = f'dqn_agent_{datetime.now().strftime("%Y_%m_%d_%H_%M")}'
    agent.save_model(f'trained_agents/{model_name}')


def evaluate(agent_type='dqn',
             num_episodes=100,
             logging_level='info',
             default_cost=500,
             trained_policy_net=None):
    logger = monopoly_logger.get_monopoly_logger(__name__, logging_level)
    logger.info(f'Starting Monopoly Game with {agent_type} agent!')

    agent = get_agent(agent_type, logging_level,
                      trained_policy_net=trained_policy_net,
                      num_episodes=num_episodes)

    episode_rewards, agent_wins = run_episodes(agent, num_episodes, logger, logging_level, default_cost, is_train=False)

    avg_rewards = [np.mean(x) for x in episode_rewards]
    plt.scatter(range(len(avg_rewards)), avg_rewards)
    plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(avg_rewards)}')


if __name__ == '__main__':
    # train(agent_type='dqn',
    #       num_episodes=10000,
    #       logging_level='info',
    #       update_target_net_freq=50,
    #       batch_size=512,
    #       default_cost=500,
    #       max_experience_len=16384,
    #       lr=0.001,
    #       hidden_layer_sizes=[64, 64])
    trained_policy_net = 'trained_agents/dqn_agent_2025_04_09_18_43'
    evaluate(agent_type='dqn',
             num_episodes=1000,
             logging_level='info',
             default_cost=500,
             trained_policy_net=trained_policy_net)
