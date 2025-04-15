from typing import List

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


def main(agent_type: str = 'baseline',
         num_episodes: int = 100,
         logging_level: str = 'info',
         update_target_net_freq: int = 5,
         batch_size: int = 512,
         default_cost: int = 500,
         max_experience_len: int = 16384,
         lr: float = 0.001,
         hidden_layer_sizes: List[int] = [256, 256, 256]):
    assert agent_type in ['random', 'baseline', 'dqn']
    logger = monopoly_logger.get_monopoly_logger(__name__, logging_level)

    logger.info(f'Starting Monopoly Game with {agent_type} agent!')

    if agent_type == 'random':
        agent = Agent(logging_level=logging_level)
    elif agent_type == 'baseline':
        agent = BaselineAgent(logging_level=logging_level)
    else:
        eps_decay = num_episodes - (num_episodes // 10)
        agent = DqnAgent(logging_level=logging_level,
                         batch_size=batch_size,
                         eps_decay=eps_decay,
                         max_experience_len=max_experience_len,
                         lr=lr,
                         update_target_net_freq=update_target_net_freq,
                         hidden_layer_sizes=hidden_layer_sizes)

    agent_wins = []
    episode_rewards = []

    for episode in range(num_episodes):
        logger.info(f'===== Starting Episode {episode} =====')
        agent.reset()
        game_end = False

        board = Board(default_cost=default_cost, logging_level=logging_level)
        step = 0
        game_rewards = []
        while not game_end:
            curr_state = board.state
            next_move = agent.get_action(curr_state)
            reward, next_state, game_end = board.execute_action(next_move, agent, step)
            agent.update_epsilon(episode)
            if agent.experience is not None:
                agent.experience.push((curr_state, next_move, reward, next_state, game_end))
            agent.optimize(episode)

            game_rewards.append(reward)
            step += 1

        episode_rewards.append(game_rewards)
        agent_won = board.opponent_monies[-1] <= 0
        agent_wins.append(agent_won)
        if agent_won:
            logger.info('===== Agent won! =====')
        else:
            logger.info('===== Opponent won! =====')
        logger.info(f'===== Game ended in {len(game_rewards)} steps =====')

        # if episode % 10 == 0:
        #     plt.plot(board.agent_monies, label='Agent Money')
        #     plt.plot(board.opponent_monies, label='Opponent Money')
        #     plt.legend()
        #     plt.show()

    average_episode_rewards = [np.mean(x) for x in episode_rewards]
    plt.scatter(x=range(len(average_episode_rewards)), y=average_episode_rewards)
    plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(average_episode_rewards)}')

    logger.info(f'Agent last 50 average win rate: {np.mean(agent_wins[-50:])}')
    logger.info(f'Agent last 50 average reward: {np.mean(average_episode_rewards[-50:])}')

    current_datetime = datetime.now().strftime('%Y_%m_%d_%H_%M')
    model_name = f'dqn_agent_{current_datetime}'
    agent.save_model(f'trained_agents/{model_name}')


def evaluate(agent_type='dqn',
             num_episodes=100,
             logging_level='info',
             default_cost=500,
             trained_policy_net=None):
    assert agent_type in ['random', 'baseline', 'dqn']
    logger = monopoly_logger.get_monopoly_logger(__name__, logging_level)

    logger.info(f'Starting Monopoly Game with {agent_type} agent!')

    if agent_type == 'random':
        agent = Agent(logging_level=logging_level)
    elif agent_type == 'baseline':
        agent = BaselineAgent(logging_level=logging_level)
    else:
        assert trained_policy_net, 'Please pass a valid trained DQN net'
        agent = DqnAgent(logging_level=logging_level)
        agent.policy_net.load_state_dict(torch.load(trained_policy_net, weights_only=True))
        agent.eps = 0

    agent_wins = []
    episode_rewards = []

    for episode in range(num_episodes):
        logger.info(f'===== Starting Episode {episode} =====')
        agent.reset()
        game_end = False

        board = Board(default_cost=default_cost, logging_level=logging_level)
        step = 0
        game_rewards = []
        while not game_end:
            curr_state = board.state
            next_move = agent.get_action(curr_state)
            reward, next_state, game_end = board.execute_action(next_move, agent, step)
            logger.info(f'Reward: {reward}')

            game_rewards.append(reward)
            step += 1

        episode_rewards.append(game_rewards)
        agent_won = board.opponent_monies[-1] <= 0
        agent_wins.append(agent_won)
        if agent_won:
            logger.info('===== Agent won! =====')
        else:
            logger.info('===== Opponent won! =====')
        logger.info(f'===== Game ended in {len(game_rewards)} steps =====')

    average_episode_rewards = [np.mean(x) for x in episode_rewards]
    plt.scatter(x=range(len(average_episode_rewards)), y=average_episode_rewards)
    plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    logger.info(f'Agent average reward: {np.mean(average_episode_rewards)}')


if __name__ == '__main__':
    main(agent_type='dqn',
         num_episodes=1000,
         logging_level='info',
         update_target_net_freq=50,
         batch_size=512,
         default_cost=500,
         max_experience_len=16384,
         lr=0.001,
         hidden_layer_sizes=[512, 512, 512])
    # trained_policy_net = 'trained_agents/dqn_agent_2025_04_09_18_43'
    # evaluate(agent_type='dqn',
    #          num_episodes=1000,
    #          logging_level='info',
    #          default_cost=500,
    #          trained_policy_net=trained_policy_net)
