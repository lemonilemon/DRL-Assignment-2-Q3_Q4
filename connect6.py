import sys
import numpy as np
import random
import copy
import math
from scipy.signal import convolve2d
from loguru import logger
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Union, Tuple

logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")


@dataclass
class Node:
    state: "Connect6Game"  # Game state
    move: Tuple[int, int, int]  # (r, c, color)
    parent: Union["Node", None] = None
    children: Dict[Tuple[int, int, int], "Node"] = field(
        default_factory=lambda: defaultdict()
    )
    visits: int = 0
    value: float = 0.0
    untried_actions: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.untried_actions = self.get_legal_actions()

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        empty_positions = [
            (r, c)
            for r in range(self.state.size)
            for c in range(self.state.size)
            if self.state.board[r, c] == 0
        ]
        return self.state.mask(empty_positions)

    def fully_expanded(self):
        return len(self.untried_actions) == 0


class MCTS:
    def __init__(
        self,
        game,
        iterations=1000,
        exploration_constant=0.141,
        rollout_depth=0,
        expansions_per_simulation=1,
    ):
        self.game = game
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.expansions_per_simulation = expansions_per_simulation
        self.root = None

    def select_child(self, node: Node) -> Union[Node, None]:
        if not node.children:
            return None
        actions = [k for k, v in node.children.items() if v is not None]
        if not actions:
            return None
        uct_values = []
        for action in actions:
            child = node.children[action]
            if child.visits == 0:
                uct_value = float("inf")
            else:
                q = child.value / child.visits
                exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
                uct_value = q + exploration
            uct_values.append((uct_value, child))
        _, best_child = max(uct_values, key=lambda x: x[0])
        return best_child

    def rollout(self, sim_game: "Connect6Game") -> float:
        current_game = sim_game.light_copy()
        my_color = sim_game.whose_turn()
        op_color = 3 - my_color
        for _ in range(self.rollout_depth):
            legal_moves = [
                (r, c)
                for r in range(sim_game.size)
                for c in range(sim_game.size)
                if current_game.board[r, c] == 0
            ]
            if not legal_moves:
                break
            r, c = random.choice(legal_moves)
            color = sim_game.whose_turn()
            current_game.board[r, c] = color
            if current_game.check_win():
                return 1.0 if color == my_color else -1.0
            current_game.board[r, c] = 0
        score = current_game.evaluate_board(my_color)
        return max(min(score, 1.0), -1.0)

    def backpropagate(self, node: Node, reward: float, color: int):
        logger.debug(f"Backpropagating: {node.move} reward={reward} color={color}")
        current = node
        while current is not None:
            current.visits += 1
            node_color = current.move[2]
            current.value += reward if node_color == color else -reward
            current = current.parent

    def run_simulation(self):
        # logger.debug("Running simulation")
        node = self.root
        sim_game = self.game.light_copy()

        # Selection
        while node.fully_expanded() and node.children:
            child = self.select_child(node)
            if child is None:
                break
            node = child
            r, c, color = node.move
            sim_game.board[r, c] = color

        # Multiple Expansions
        if node.untried_actions:
            # Expand up to expansions_per_simulation actions
            num_expansions = min(self.expansions_per_simulation, len(node.untried_actions))
            actions = random.sample(node.untried_actions, num_expansions)
            new_nodes = []
            for action in actions:
                node.untried_actions.remove(action)
                r, c = action
                color = sim_game.whose_turn()
                sim_game.board[r, c] = color
                new_node = Node(state=sim_game.light_copy(), move=(r, c, color), parent=node)
                node.children[(r, c, color)] = new_node
                sim_game.board[r, c] = 0  # Reset the board for the next expansion
                new_nodes.append(new_node)

            # Rollout from each new node
            for new_node in new_nodes:
                rollout_reward = self.rollout(new_node.state)
                self.backpropagate(new_node, rollout_reward, new_node.state.whose_turn())

    def best_action_distribution(self):
        total_visits = sum(
            child.visits for child in self.root.children.values() if child is not None
        )
        distribution = {}
        best_visits = -1
        best_value = -1
        best_action = None
        for action, child in self.root.children.items():
            if child is None:
                continue
            distribution[action] = (
                child.visits / total_visits if total_visits > 0 else 0
            )
            logger.debug(
                f"Action {action}: visits={child.visits}, avg_value={child.value / child.visits if child.visits > 0 else 0:.3f}"
            )
            if child.visits > best_visits:
                best_visits = child.visits
                best_value = child.value / child.visits if child.visits > 0 else 0
                best_action = action
            elif child.visits == best_visits and child.value / child.visits > best_value:
                best_visits = child.visits
                best_value = child.value / child.visits if child.visits > 0 else 0
                best_action = action
                
        return best_action, distribution



class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.game_over = False
        self.last_opponent_move = None
        self.MCT = MCTS(self)
        # Precompute positional bias
        center = size // 2
        self.positional_bias = np.zeros((size, size), dtype=np.float32)
        for r in range(size):
            for c in range(size):
                distance = abs(r - center) + abs(c - center)
                self.positional_bias[r, c] = max(0, (size - distance) / size) * 2

    def light_copy(self):
        new_game = Connect6Game(self.size)
        new_game.board = self.board.copy()
        new_game.game_over = self.game_over
        self.last_opponent_move = self.last_opponent_move
        self.positional_bias = self.positional_bias.copy()
        # Don't copy the MCT tree
        return new_game

    def reset_board(self):
        self.board.fill(0)
        self.game_over = False
        self.MCT = MCTS(self)
        print("= ", flush=True)

    def set_board_size(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.game_over = False
        self.MCT = MCTS(self)
        center = size // 2
        self.positional_bias = np.zeros((size, size), dtype=np.float32)
        for r in range(size):
            for c in range(size):
                distance = abs(r - center) + abs(c - center)
                self.positional_bias[r, c] = max(0, (size - distance) / size) * 2
        print("= ", flush=True)

    def check_win(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if (
                            0 <= prev_r < self.size
                            and 0 <= prev_c < self.size
                            and self.board[prev_r, prev_c] == current_color
                        ):
                            continue
                        count = 0
                        rr, cc = r, c
                        while (
                            0 <= rr < self.size
                            and 0 <= cc < self.size
                            and self.board[rr, cc] == current_color
                        ):
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            self.game_over = True
                            return current_color
        return 0

    def index_to_label(self, col):
        return chr(ord("A") + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        col_char = col_char.upper()
        if col_char >= "J":
            return ord(col_char) - ord("A") - 1
        return ord(col_char) - ord("A")

    def play_move(self, color, move):
        logger.debug(f"Play move: {color} {move}")
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(",")
        positions = []
        for stone in stones:
            stone = stone.strip()
            col_char = stone[0].upper()
            col = self.label_to_index(col_char)
            row = int(stone[1:]) - 1
            if (
                not (0 <= row < self.size and 0 <= col < self.size)
                or self.board[row, col] != 0
            ):
                logger.debug(
                    f"Invalid move: {stone} {row} {col} {self.board[row, col]}"
                )
                print("? Invalid move")
                return
            positions.append((row, col))
        
        game_color = 1 if color.upper() == "B" else 2
        for row, col in positions:
            self.board[row, col] = game_color

            # Track if we made a change to MCT root 
            root_updated = False
            
            # Try to find the move in the existing tree
            if self.MCT.root is not None and (row, col, game_color) in self.MCT.root.children:
                self.MCT.root = self.MCT.root.children[(row, col, game_color)]
                self.MCT.root.parent = None  # Detach from parent
                root_updated = True
            
            # If we couldn't update the root through the tree, create a new root with the current state
            if not root_updated or self.MCT.root is None:
                self.MCT.root = Node(state=self.light_copy(), move=(0, 0, 0))

        self.last_opponent_move = positions[-1]
        if self.check_win():
            self.game_over = True
        print("= ", end="", flush=True)
        return

    def mask(self, empty_positions):
        points = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board[r, c] != 0
        ]
        if not points:
            return empty_positions
        min_x = max_x = points[0][0]
        min_y = max_y = points[0][1]
        for p in points:
            x, y = p[0], p[1]
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        ret = []
        for ep in empty_positions:
            if min_x - 2 <= ep[0] <= max_x + 2 and min_y - 2 <= ep[1] <= max_y + 2:
                ret.append(ep)
        return ret

    @logger.catch
    def generate_move(self, color):
        if self.game_over:
            print("? Game over", flush=True)
            return

        # First move
        if self.MCT.root is None:
            self.MCT.root = Node(state=self.light_copy(), move=(0, 0, 0))
        logger.debug("START SIMULATE FIRST MOVE")
        for _ in range(self.MCT.iterations):
            self.MCT.run_simulation()
        best_action, _ = self.MCT.best_action_distribution()
        if best_action is None:
            print("? No legal moves", flush=True)
            return
        r1, c1, _ = best_action
        # moves.append((r1, c1))

        # Update board with first move
        move_str = f"{self.index_to_label(c1)}{r1 + 1}"
        self.play_move(color, move_str)
        print(f"{move_str}\n", flush=True)

        return

    def evaluate_board(self, my_color: int):
        """Enhanced board evaluation with positional bias and threat detection"""
        # Local variables
        cs = np.array([0, 1, 10, 100, 1000, 10000, 100000], dtype=np.float32)
        directions = [
            ('horizontal', np.ones((1, 6), dtype=np.int8)),
            ('vertical', np.ones((6, 1), dtype=np.int8)),
            ('diagonal', np.eye(6, dtype=np.int8)),
            ('anti-diagonal', np.fliplr(np.eye(6, dtype=np.int8)))
        ]
        score = {1: 0.0, 2: 0.0}

        # Step 1: Positional bias (vectorized)
        black_mask = (self.board == 1)
        white_mask = (self.board == 2)
        score[1] += np.sum(self.positional_bias[black_mask])
        score[2] += np.sum(self.positional_bias[white_mask])
        opponent_color = 3 - my_color

        # Step 2: Sequence evaluation using convolution
        for color in [opponent_color, my_color]:
            # Binary masks for this color and opponent
            color_mask = (self.board == color).astype(np.int8)
            opponent = 3 - color
            opponent_mask = (self.board == opponent).astype(np.int8)
            
            for direction, kernel in directions:
                # Count your stones in 6-position windows
                conv = convolve2d(color_mask, kernel, mode='valid')
                # Count opponent stones in the same windows
                opp_conv = convolve2d(opponent_mask, kernel, mode='valid')
                
                # Score sequences based on length
                for length in range(1, 7):
                    seq_mask = (conv == length)
                    if not np.any(seq_mask):
                        continue
                    if length == 6:
                        return 1.0 if color == my_color else -1.0
                    # Count sequences with no opponent stones (opp_conv == 0)
                    valid_seq_mask = seq_mask & (opp_conv == 0)
                    if length < 6 and not np.any(valid_seq_mask):
                        continue
                    # Add number of valid sequences to score
                    valid_count = np.sum(valid_seq_mask)
                    score[color] += valid_count * cs[length]


        return (score[my_color] - score[opponent_color]) / (score[my_color] + score[opponent_color])

    def show_board(self):
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row + 1:2} " + " ".join(
                "X"
                if self.board[row, col] == 1
                else "O"
                if self.board[row, col] == 2
                else "."
                for col in range(self.size)
            )
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        print("= ", flush=True)

    def process_command(self, command):
        logger.debug(f"Process command: {command.strip()}")
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print(f"env_board_size={self.size}", flush=True)
        if not command:
            return
        parts = command.split()
        cmd = parts[0].lower()
        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

    def whose_turn(self):
        black_num = np.count_nonzero(self.board == 1)
        white_num = np.count_nonzero(self.board == 2)
        black_turn_done = (black_num + 1) // 2
        white_turn_done = white_num // 2
        return 1 if black_turn_done <= white_turn_done else 2


if __name__ == "__main__":
    game = Connect6Game()
    game.run()
