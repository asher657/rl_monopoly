import monopoly_logger
from agent import Agent
from board import Board
from baseline_agent import BaselineAgent

import matplotlib.pyplot as plt
import numpy as np


def main(agent_type: str = 'baseline', num_episodes: int = 100, logging_level: str = 'info'):
    assert agent_type in ['random', 'baseline', 'dqn']
    logger = monopoly_logger.get_monopoly_logger(__name__, logging_level)

    logger.info(f'Starting Monopoly Game with {agent_type} agent!')

    agent_wins = []
    for episode in range(num_episodes):
        logger.info(f'===== Starting Episode {episode} =====')
        game_end = False

        if agent_type == 'random':
            agent = Agent(logging_level=logging_level)
        elif agent_type == 'baseline':
            agent = BaselineAgent(logging_level=logging_level)
        else:
            agent = Agent(logging_level=logging_level)  # TODO replace with dqn agent

        board = Board(default_cost=0, logging_level=logging_level)
        step = 0
        while not game_end:
            next_move = agent.get_action(board.state)
            reward, next_state, game_end = board.execute_action(next_move, agent, step)
            step += 1

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
    main('baseline', 200, 'info')
