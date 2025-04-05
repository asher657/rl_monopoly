from agent import Agent
from board import Board
from baseline_agent import BaselineAgent

import matplotlib.pyplot as plt


def main():
    LOGGING_LEVEL = 'debug'

    for episode in range(1):
        game_end = False
        print(f'=== Episode {episode + 1} ===')
        # agent = BaselineAgent(logging_level=LOGGING_LEVEL)
        agent = Agent(logging_level=LOGGING_LEVEL)
        board = Board(default_cost=0, logging_level=LOGGING_LEVEL)
        step = 0
        while not game_end:
            next_move = agent.get_action(board.state)
            reward, next_state, game_end = board.execute_action(next_move, agent, step)
            step += 1

        plt.plot(board.agent_monies, label='Agent Money')
        plt.plot(board.opponent_monies, label='Opponent Money')
        plt.legend()
        plt.show()




if __name__ == '__main__':
    main()
