import gym
from gym import spaces
import numpy as np
from connect6 import Connect6Game  # adjust the import as needed

class Connect6GymEnv(gym.Env):
    """
    A custom Gym environment wrapping the Connect6Game.
    The agent (Black) selects a move (a single board position) 
    and then the opponent (White) makes a random move.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=19):
        super().__init__()
        self.board_size = board_size
        self.game = Connect6Game(size=board_size)
        
        # Observation space: a board with values {0: empty, 1: black, -1: white}
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(board_size, board_size),
            dtype=np.int32
        )
        
        # Action space: Discrete number representing a position on the board (row-major indexing)
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # A flag to indicate terminal state.
        self.done = False

    def reset(self):
        """Resets the game state and returns the initial observation."""
        self.game.reset_board()
        self.done = False
        # Return a copy of the board state
        return self.game.board.copy()

    def step(self, action):
        """
        Executes one time step within the environment.
        The action is an integer corresponding to a board cell in row-major order.
        """
        if self.done:
            return self.game.board.copy(), 0.0, self.done, {"msg": "Game over"}

        row = action // self.board_size
        col = action % self.board_size

        # Check if the selected cell is already occupied
        if self.game.board[row, col] != 0:
            # Illegal move: penalize and end the episode
            reward = -1.0
            self.done = True
            return self.game.board.copy(), reward, self.done, {"illegal_move": True}

        # Format the move string as expected by the Connect6Game code
        move_str = f"{self.game.index_to_label(col)}{row+1}"
        
        # Agent (Black) makes its move.
        self.game.play_move('B', move_str)
        winner = self.game.check_win()
        if winner == 1:
            reward = 1.0
            self.done = True
            return self.game.board.copy(), reward, self.done, {"result": "win"}
        elif winner == -1:
            reward = -1.0
            self.done = True
            return self.game.board.copy(), reward, self.done, {"result": "loss"}

        # Environment (opponent, White) makes a random move.
        self.game.generate_move('W')
        winner = self.game.check_win()
        if winner == 2:
            reward = -1.0
            self.done = True
            info = {"result": "loss"}
        elif winner == 1:
            reward = 1.0
            self.done = True
            info = {"result": "win"}
        else:
            reward = 0.0
            info = {}

        return self.game.board.copy(), reward, self.done, info

    def render(self, mode="human"):
        """Renders the current game board."""
        self.game.show_board()

    def close(self):
        """Clean up resources if needed."""
        pass

# Example usage:
if __name__ == "__main__":
    env = Connect6GymEnv(board_size=9)  # smaller board for quick testing
    obs = env.reset()
    env.render()
    
    # Sample a random legal move (this does not ensure legality in general)
    action = env.action_space.sample()
    print("Taking action:", action)
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"Reward: {reward}, Done: {done}, Info: {info}")

