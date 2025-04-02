from board import Board
from agent import Agent
from baseline_agent import BaselineAgent


def main():
    LOGGING_LEVEL = 'debug'

    for episode in range(1):
        game_end = False
        print(f'=== Episode {episode + 1} ===')
        agent = BaselineAgent(logging_level=LOGGING_LEVEL)
        board = Board(logging_level=LOGGING_LEVEL)
        step = 0
        while not game_end:
            next_move = agent.get_action(board.state)
            reward, next_state, game_end = board.execute_action(next_move, agent.money, step)
            step += 1

        board.rewards
        board.successes


if __name__ == '__main__':
    main()
