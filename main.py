from board import Board
from agent import Agent

def main():
    LOGGING_LEVEL = 'debug'
    agent = Agent(logging_level=LOGGING_LEVEL)
    board = Board(logging_level=LOGGING_LEVEL)
    # next_move = agent.get_action(board.state)
    next_move = 25
    reward, next_state, game_end = board.execute_action(next_move, agent.money)
    print(reward, game_end)

if __name__ == '__main__':
    main()