import monopoly_logger
from agent import Agent
from board import Board
from baseline_agent import BaselineAgent

import matplotlib.pyplot as plt
import numpy as np
import os

from dqnagent import DqnAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(agent_type: str = 'baseline', num_episodes: int = 100, logging_level: str = 'info', optimize_param: int = 5):
    assert agent_type in ['random', 'baseline', 'dqn']
    logger = monopoly_logger.get_monopoly_logger(__name__, logging_level)

    logger.info(f'Starting Monopoly Game with {agent_type} agent!')

    if agent_type == 'random':
        agent = Agent(logging_level=logging_level)
    elif agent_type == 'baseline':
        agent = BaselineAgent(logging_level=logging_level)
    else:
        eps_decay = num_episodes - (num_episodes // 10)
        agent = DqnAgent(logging_level=logging_level, batch_size=256, eps_decay=eps_decay)

    agent_wins = []
    episode_rewards = []

    for episode in range(num_episodes):
        logger.info(f'===== Starting Episode {episode} =====')
        agent.reset()
        game_end = False

        board = Board(default_cost=500, logging_level=logging_level)
        step = 0
        game_rewards = []
        while not game_end:
            curr_state = board.state
            next_move = agent.get_action(curr_state)
            reward, next_state, game_end = board.execute_action(next_move, agent, step)
            agent.update_epsilon(episode)
            if agent.experience is not None:
                agent.experience.push((curr_state, next_move, reward, next_state, game_end))
            agent.optimize()

            game_rewards.append(reward)
            step += 1

        if episode % optimize_param == 0:
            agent.update_target_network()

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

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')
    average_episode_rewards = [np.mean(x) for x in episode_rewards]
    plt.scatter(x=range(len(average_episode_rewards)), y=average_episode_rewards)
    plt.show()


if __name__ == '__main__':
    main('dqn', 10000, 'info', optimize_param=100)
