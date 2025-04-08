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
        agent = DqnAgent(logging_level=logging_level, batch_size=10)

    agent_wins = []
    for episode in range(num_episodes):
        logger.info(f'===== Starting Episode {episode} =====')
        agent.reset()
        game_end = False

        board = Board(default_cost=0, logging_level=logging_level)
        step = 0
        while not game_end:
            curr_state = board.state
            next_move = agent.get_action(curr_state)
            reward, next_state, game_end = board.execute_action(next_move, agent, step)
            agent.update_epsilon(episode)
            if agent.experience:
                agent.experience.push((curr_state, next_move, reward, next_state, game_end))
            agent.optimize()
            step += 1
        if episode % optimize_param == 0:
            agent.update_target_network()

        agent_won = board.opponent_monies[-1] <= 0
        agent_wins.append(agent_won)
        if agent_won:
            logger.info('===== Agent won! =====')
        else:
            logger.info('===== Opponent won! =====')

        if episode % 10 == 0:
            plt.plot(board.agent_monies, label='Agent Money')
            plt.plot(board.opponent_monies, label='Opponent Money')
            plt.legend()
            plt.show()

    logger.info(f'Agent average win rate: {np.mean(agent_wins)}')


if __name__ == '__main__':
    main('dqn', 500, 'info')
