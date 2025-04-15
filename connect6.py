import sys
import numpy as np
import random
import copy
import math
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
        iterations=100,
        exploration_constant=0.001,
        rollout_depth=0,
        expansions_per_simulation=5,
    ):
        self.game = game
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.expansions_per_simulation = expansions_per_simulation

    def create_game_from_board(self, board):
        new_game = copy.deepcopy(self.game)
        new_game.board = board.copy()
        return new_game

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
        current_game = copy.deepcopy(sim_game)
        my_color = 1 if self.game.turn == 1 else 2
        op_color = 3 - my_color
        for _ in range(self.rollout_depth):
            legal_moves = [
                (r, c)
                for r in range(current_game.size)
                for c in range(current_game.size)
                if current_game.board[r, c] == 0
            ]
            if not legal_moves:
                break
            r, c = random.choice(legal_moves)
            color = current_game.whose_turn()
            current_game.board[r, c] = color
            if current_game.check_win():
                return 10.0 if color == my_color else -10.0
        score = current_game.evaluate_board()
        value = (score[my_color] - score[op_color]) / 10000
        return max(min(value, 10.0), -10.0)

    def backpropagate(self, node: Node, reward: float):
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def run_simulation(self, root: Node):
        node = root
        sim_game = copy.deepcopy(self.game)

        # Selection
        while node.fully_expanded() and node.children:
            child = self.select_child(node)
            if child is None:
                break
            node = child
            r, c, color = node.move
            sim_game.board[r, c] = color

        # Prioritized Expansion
        if node.untried_actions:
            # Calculate RAVE values or quick evaluations for actions
            action_values = {}
            color = sim_game.whose_turn()
            
            # Start with a subset of actions for efficiency
            eval_candidates = node.untried_actions
            if len(eval_candidates) > 25:
                eval_candidates = random.sample(node.untried_actions, 25)
            
            # Quickly evaluate each action
            for action in eval_candidates:
                r, c = action
                
                # Place stone and evaluate
                sim_game.board[r, c] = color
                
                # Use pattern recognition or simpler eval
                value = sim_game.evaluate_board()
                action_values[action] = value
                
                # Reset board
                sim_game.board[r, c] = 0
            
            # Sort actions by value
            sorted_actions = sorted(action_values.items(), key=lambda x: x[1], reverse=True)
            
            # Select best actions to expand
            num_to_expand = min(self.expansions_per_simulation, len(sorted_actions))
            
            # Expand nodes
            expanded_nodes = []
            for i in range(num_to_expand):
                action, _ = sorted_actions[i]
                # Remove from untried
                node.untried_actions.remove(action)
                
                r, c = action
                sim_game.board[r, c] = color
                
                # Create new node
                new_node = Node(state=copy.deepcopy(sim_game), move=(r, c, color), parent=node)
                node.children[(r, c, color)] = new_node
                expanded_nodes.append(new_node)
                
                # Reset for next expansion
                sim_game.board[r, c] = 0
            
            # Do rollout and backpropagation for expanded nodes
            for expanded_node in expanded_nodes:
                # Set up board for rollout
                r, c, color = expanded_node.move
                sim_game.board[r, c] = color
                
                # Do rollout
                rollout_reward = self.rollout(expanded_node.state)
                
                # Reset
                sim_game.board[r, c] = 0
                
                # Backpropagate
                self.backpropagate(expanded_node, rollout_reward)
        # Multiple Expansions

        # if node.untried_actions:
        #     # Expand up to expansions_per_simulation actions
        #     num_expansions = min(self.expansions_per_simulation, len(node.untried_actions))
        #     actions = random.sample(node.untried_actions, num_expansions)
        #     new_nodes = []
        #     for action in actions:
        #         node.untried_actions.remove(action)
        #         r, c = action
        #         color = sim_game.whose_turn()
        #         sim_game.board[r, c] = color
        #         new_node = Node(state=copy.deepcopy(sim_game), move=(r, c, color), parent=node)
        #         node.children[(r, c, color)] = new_node
        #         sim_game.board[r, c] = 0  # Reset the board for the next expansion
        #         new_nodes.append(new_node)
        #
        #     # Rollout from each new node
        #     for new_node in new_nodes:
        #         rollout_reward = self.rollout(new_node.state)
        #         self.backpropagate(new_node, rollout_reward)
        #

        # # Expansion
        # if node.untried_actions:
        #     action = random.choice(node.untried_actions)
        #     node.untried_actions.remove(action)
        #     r, c = action
        #     color = sim_game.whose_turn()
        #     sim_game.board[r, c] = color
        #     new_node = Node(
        #         state=copy.deepcopy(sim_game), move=(r, c, color), parent=node
        #     )
        #     node.children[(r, c, color)] = new_node
        #     node = new_node
        #
        # # Rollout
        # rollout_reward = self.rollout(node.state)
        # # Backpropagate
        # self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root: Node):
        total_visits = sum(
            child.visits for child in root.children.values() if child is not None
        )
        distribution = {}
        best_visits = -1
        best_value = -1
        best_action = None
        for action, child in root.children.items():
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
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.last_opponent_move = None
        self.MCT = MCTS(self)

    def reset_board(self):
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        self.MCT = MCTS(self)
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

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == "B" else 2

        self.last_opponent_move = positions[-1]
        self.turn = self.whose_turn()
        if self.check_win():
            self.game_over = True
        print("= ", flush=True)

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

        num_stones = (
            1 if color.upper() == "B" and np.count_nonzero(self.board == 1) == 0 else 2
        )
        moves = []

        # First move
        root = Node(state=copy.deepcopy(self), move=(0, 0, 0))
        logger.debug("START SIMULATE FIRST MOVE")
        for _ in range(self.MCT.iterations):
            self.MCT.run_simulation(root)
        best_action, _ = self.MCT.best_action_distribution(root)
        if best_action is None:
            print("? No legal moves", flush=True)
            return
        r1, c1, _ = best_action
        moves.append((r1, c1))

        # Update board with first move
        game_color = 1 if color.upper() == "B" else 2
        self.board[r1, c1] = game_color

        # Second move (if needed)
        if num_stones == 2:
            # Create new root with updated board
            root = Node(state=copy.deepcopy(self), move=(0, 0, 0))
            logger.debug("START SIMULATE SECOND MOVE")
            for _ in range(self.MCT.iterations):
                self.MCT.run_simulation(root)
            best_action, _ = self.MCT.best_action_distribution(root)
            if best_action is None:
                print("? No legal moves", flush=True)
                # Revert first move if second fails
                self.board[r1, c1] = 0
                return
            r2, c2, _ = best_action
            # Ensure distinct moves
            if (r2, c2) == (r1, c1):
                print("? Second move cannot be the same as first", flush=True)
                self.board[r1, c1] = 0
                return
            moves.append((r2, c2))

        # Apply moves via play_move to ensure validation
        move_str = ",".join(f"{self.index_to_label(c)}{r + 1}" for r, c in moves)
        # Temporarily revert board for play_move
        self.board[r1, c1] = 0
        self.play_move(color, move_str)
        print(move_str, flush=True)
        return

    def evaluate_board(self):
        """Enhanced board evaluation with positional bias and threat detection"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = {1: 0, 2: 0}
        
        # Sequence values (exponential scaling)
        cs = {0: 0, 1: 1, 2: 10, 3: 100, 4: 1000, 5: 10000, 6: 100000}
        
        # Positional bias - center is better
        center = self.size // 2
        
        # Check for sequences in all directions
        for r in range(self.size):
            for c in range(self.size):
                # Add positional bias
                if self.board[r, c] != 0:
                    color = self.board[r, c]
                    # Distance from center (Manhattan distance)
                    distance = abs(r - center) + abs(c - center)
                    position_value = max(0, (self.size - distance) / self.size)
                    score[color] += position_value * 2  # Small bonus for central positions
                
                # Check sequences
                for dr, dc in directions:
                    counter = [0, 0, 0]  # [empty, black, white]
                    blocks = [0, 0]      # Blocked ends [start, end]
                    rr, cc = r, c
                    
                    # Check if starting point is blocked
                    if not (0 <= r-dr < self.size and 0 <= c-dc < self.size):
                        blocks[0] = 1
                    elif self.board[r-dr, c-dc] != 0:
                        if rr < self.size and cc < self.size and self.board[rr, cc] != 0 and self.board[rr, cc] != self.board[r-dr, c-dc]:
                            blocks[0] = 1
                    
                    # Count stones in sequence
                    for i in range(6):
                        if 0 <= rr < self.size and 0 <= cc < self.size:
                            counter[self.board[rr, cc]] += 1
                            rr += dr
                            cc += dc
                        else:
                            blocks[1] = 1
                            break
                    
                    # Check if end point is blocked
                    if 0 <= rr < self.size and 0 <= cc < self.size:
                        if self.board[rr, cc] != 0:
                            if self.board[rr, cc] != self.board[r, c] and self.board[r, c] != 0:
                                blocks[1] = 1
                    else:
                        blocks[1] = 1
                    
                    # Evaluate the sequence
                    if sum(counter) == 6:
                        # Pure sequences
                        if counter[1] > 0 and counter[2] == 0:
                            # Black sequence
                            value = cs[counter[1]]
                            # Reduce score if blocked
                            if blocks[0] + blocks[1] == 1:  # One end blocked
                                value *= 0.8
                            elif blocks[0] + blocks[1] == 2:  # Both ends blocked
                                value *= 0.5
                            score[1] += value
                            
                        elif counter[2] > 0 and counter[1] == 0:
                            # White sequence
                            value = cs[counter[2]]
                            # Reduce score if blocked
                            if blocks[0] + blocks[1] == 1:  # One end blocked
                                value *= 0.8
                            elif blocks[0] + blocks[1] == 2:  # Both ends blocked
                                value *= 0.5
                            score[2] += value
        
        # Detect immediate threats (opponent about to win)
        for color in [1, 2]:
            opponent = 3 - color
            # Check if opponent has a winning threat
            for r in range(self.size):
                for c in range(self.size):
                    if self.board[r, c] == 0:
                        # Try opponent's move
                        self.board[r, c] = opponent
                        if self.check_win():
                            # Opponent has a winning move
                            score[color] -= 8000  # Heavy penalty
                        self.board[r, c] = 0  # Reset
        return score

    def evaluate_position(self, r, c, color):
        self.board[r, c] = color
        score = self.evaluate_board()
        self.board[r, c] = 0
        return score[color]

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
        if black_num == 0 and white_num == 0:
            return 1  # Black starts
        if black_num == 1 and white_num == 0:
            return 2
        expected_black = (white_num * 2) + 1
        if black_num == expected_black:
            return 2
        return 1


if __name__ == "__main__":
    game = Connect6Game()
    game.run()
