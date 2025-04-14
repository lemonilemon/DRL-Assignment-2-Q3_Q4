import copy
import random
import math
import numpy as np

# class TD_MCTS_Node:
#     def __init__(self, env, state, score, parent=None, action=None, probability=1.0, is_max_node=True):
#         self.state = state
#         self.score = score
#         self.parent = parent
#         self.action = action
#         self.is_max_node = is_max_node
#         self.probability = probability
#         self.children = {}
#         self.visits = 0
#         self.total_reward = 0.0
#         if is_max_node:
#             self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
#         else:
#             # For chance nodes, untried actions represent possible tile placements
#             self.untried_actions = self.get_chance_actions(state) if state is not None else []
#
#     def get_chance_actions(self, state):
#         # Generate unique actions for each possible tile placement
#         actions = []
#         empty_cells = [(i, j) for i in range(state.shape[0]) for j in range(state.shape[1]) if state[i, j] == 0]
#         for i, j in empty_cells:
#             for value in [2, 4]:  # Possible tile values
#                 actions.append((i, j, value))
#         return actions
#
#     def fully_expanded(self):
#         return len(self.untried_actions) == 0
#
# class TD_MCTS:
#     def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=50, gamma=0.99):
#         self.env = env
#         self.approximator = approximator
#         self.iterations = iterations
#         self.c = exploration_constant
#         self.rollout_depth = rollout_depth
#         self.gamma = gamma
#
#     def create_env_from_state(self, state, score):
#         new_env = copy.deepcopy(self.env)
#         new_env.board = state.copy()
#         new_env.score = score
#         return new_env
#
#     def select_child(self, node):
#         if not node.children:
#             return None
#
#         if node.is_max_node:
#             # Max node: Use UCT for selection
#             uct_values = []
#             for action, child in node.children.items():
#                 if child.visits == 0:
#                     uct_value = float('inf')
#                 else:
#                     exploitation = child.total_reward / child.visits
#                     exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
#                     uct_value = exploitation + exploration
#                 uct_values.append((uct_value, action))
#             _, best_action = max(uct_values)
#             return best_action
#         else:
#             # Chance node: Select a child randomly based on probabilities
#             if not node.children:
#                 return None
#             actions = list(node.children.keys())
#             probabilities = [child.probability for child in node.children.values()]
#             return random.choices(actions, weights=probabilities, k=1)[0]
#
#     def expand_chance_node(self, node, sim_env):
#         if not node.untried_actions:
#             return None
#
#         # Pick an untried tile placement
#         i, j, value = node.untried_actions.pop(0)
#         new_state = node.state.copy()
#         new_state[i, j] = value
#         empty_cells = len([(r, c) for r in range(sim_env.board.shape[0])
#                           for c in range(sim_env.board.shape[1]) if node.state[r, c] == 0])
#         prob = (0.9 if value == 2 else 0.1) / empty_cells
#
#         # Update sim_env to reflect the new state
#         sim_env = self.create_env_from_state(new_state, node.score)
#
#         # Create a max node as child with tuple action_id
#         action_id = (i, j, value)  # Use tuple instead of string
#         node.children[action_id] = TD_MCTS_Node(sim_env, new_state, node.score,
#                                                is_max_node=True, parent=node,
#                                                action=action_id, probability=prob)
#         return node.children[action_id]
#
#     def rollout(self, sim_env, depth):
#         current_env = copy.deepcopy(sim_env)  # Avoid modifying the original sim_env
#         total_reward = 0.0
#         discount = 1.0
#
#         for _ in range(depth):
#             legal_moves = [a for a in range(4) if current_env.is_move_legal(a)]
#             if not legal_moves:  # Game over
#                 break
#
#             # Max node: Choose a random legal move (simulating player's decision)
#             action = random.choice(legal_moves)
#             prev_score = current_env.score
#             _, new_score, done, _ = current_env.step(action)
#             total_reward += discount * (new_score - prev_score)
#             if done:
#                 break
#
#             # Chance node: Simulate random tile placement
#             empty_cells = [(i, j) for i in range(current_env.board.shape[0])
#                           for j in range(current_env.board.shape[1]) if current_env.board[i, j] == 0]
#             if not empty_cells:
#                 break
#
#             # Randomly select a cell and tile value based on probabilities
#             i, j = random.choice(empty_cells)
#             value = random.choices([2, 4], weights=[0.9, 0.1], k=1)[0]
#             current_env.board[i, j] = value
#             discount *= self.gamma
#
#         # Evaluate the final state using the approximator
#         final_value = self.approximator.value(current_env.board)
#         return total_reward + discount * final_value
#
#     def run_simulation(self, root):
#         node = root
#         sim_env = self.create_env_from_state(node.state, node.score)
#         path = []
#
#         # Selection: Traverse until a non-fully expanded node or leaf
#         while node.fully_expanded() and node.children:
#             action = self.select_child(node)
#             if action is None:
#                 break
#             path.append((node, action))
#             if node.is_max_node:
#                 # Apply player's move
#                 state, _, done, _ = sim_env.step(action)
#                 if done:
#                     reward = sim_env.score
#                     self.backpropagate(node, reward)
#                     return
#                 # Update sim_env to the new state
#                 sim_env = self.create_env_from_state(state, sim_env.score)
#                 # Move to chance node
#                 if action not in node.children:
#                     # Create chance node after player's move
#                     node.children[action] = TD_MCTS_Node(sim_env, state, sim_env.score,
#                                                         is_max_node=False, parent=node, action=action)
#                 node = node.children[action]
#             else:
#                 # Chance node: Move to next max node
#                 node = node.children[action]
#                 # Update sim_env to the child's state
#                 sim_env = self.create_env_from_state(node.state, node.score)
#
#         # Expansion
#         if node.untried_actions:
#             if node.is_max_node:
#                 # Expand max node
#                 action = random.choice(node.untried_actions)
#                 node.untried_actions.remove(action)
#                 state, _, done, _ = sim_env.step(action)
#                 if done:
#                     reward = sim_env.score
#                     self.backpropagate(node, reward)
#                     return
#                 # Update sim_env to the new state
#                 sim_env = self.create_env_from_state(state, sim_env.score)
#                 node.children[action] = TD_MCTS_Node(sim_env, state, sim_env.score,
#                                                     is_max_node=False, parent=node, action=action)
#                 node = node.children[action]
#             else:
#                 # Expand chance node
#                 node = self.expand_chance_node(node, sim_env)
#                 if node is None:
#                     return
#                 # sim_env already updated in expand_chance_node
#                 sim_env = self.create_env_from_state(node.state, node.score)
#
#         # Rollout
#         rollout_reward = self.rollout(sim_env, self.rollout_depth)
#         # Backpropagate
#         self.backpropagate(node, rollout_reward)
#
#
#     def backpropagate(self, node, reward):
#         current = node
#         while current is not None:
#             current.visits += 1
#             if current.is_max_node:
#                 # Max node: Update with the reward
#                 current.total_reward += reward
#             else:
#                 # Chance node: Update with expected value (weighted by probability)
#                 # This is handled implicitly as children propagate their values
#                 current.total_reward += reward * current.probability
#             reward *= self.gamma
#             current = current.parent
#
#     def best_action_distribution(self, root):
#         total_visits = sum(child.visits for child in root.children.values())
#         distribution = np.zeros(4)
#         best_visits = -1
#         best_action = None
#         for action, child in root.children.items():
#             if isinstance(action, int):  # Only consider player actions (integers)
#                 distribution[action] = child.visits / total_visits if total_visits > 0 else 0
#                 if child.visits > best_visits:
#                     best_visits = child.visits
#                     best_action = action
#         return best_action, distribution

# From class assignment's Q1, Q2
import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, env, state, score, parent=None, action=None, probability=1.0, is_max_node=True):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.is_max_node = is_max_node
        self.probability = probability
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        if is_max_node:
            self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        else:
            self.untried_actions = [0] if state is not None else []

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    # def select_child(self, node):
    #     if not node.children:
    #         return None
    #
    #     if node.is_max_node:
    #         # For max nodes, use standard UCT formula
    #         uct_values = []
    #         for action, child in node.children.items():
    #             if child.visits == 0:
    #                 uct_value = float('inf')
    #             else:
    #                 exploitation = child.total_reward / child.visits
    #                 exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
    #                 uct_value = exploitation + exploration
    #             uct_values.append((uct_value, action))
    #         # Select action with highest UCT value
    #         _, best_action = max(uct_values)
    #         return best_action
    #     else:
    #         # For chance nodes, select based on probability (random sampling)
    #         # This simulates the stochastic nature of tile placements
    #         if node.children:
    #             return random.choice(list(node.children.keys()))
    #         return None

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        if not node.children:
            print(node.state, node.score, node.untried_actions, node.children)
            return None  # No children to select from

        # Calculate UCT values for all children
        uct_values = []
        for action, child in node.children.items():
            if child.visits == 0:  # Handle unvisited nodes
                uct_value = float('inf')  # Encourage exploration of unvisited nodes
            else:
                # UCT formula: exploitation (Q) + exploration term
                exploitation = child.total_reward / child.visits
                exploration = self.c * math.sqrt(math.log(node.visits) / child.visits)
                uct_value = exploitation + exploration
            uct_values.append((uct_value, action))

        # Select the action with the highest UCT value
        _, best_action = max(uct_values)
        return best_action

    def expand_chance_node(self, node, sim_env):
        # Clear placeholder
        node.untried_actions = []
        
        # Generate possible next states (in 2048, this would be placing 2 or 4 in empty cells)
        empty_cells = [(i, j) for i in range(sim_env.board.shape[0])
                      for j in range(sim_env.board.shape[1]) if sim_env.board[i, j] == 0]
        
        # For each empty cell, create children with probabilities
        for i, j in empty_cells:
            for value, prob in [(2, 0.9), (4, 0.1)]:  # 90% chance for 2, 10% for 4
                action_id = len(node.children)  # Use a unique identifier
                new_state = node.state.copy()
                new_state[i, j] = value
                child_prob = prob / len(empty_cells)
                
                # Create a max node as child
                node.children[action_id] = TD_MCTS_Node(sim_env, new_state, node.score, 
                                                      is_max_node=True, parent=node,
                                                      action=action_id, probability=child_prob)

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.
        current_env = copy.deepcopy(sim_env)  # Avoid modifying the original sim_env
        prev_score = current_env.score
        for _ in range(depth):
            legal_moves = [a for a in range(4) if current_env.is_move_legal(a)]
            if not legal_moves:  # Game over
                break
            action = random.choice(legal_moves)
            _, _, done, _ = current_env.step(action)
            if done:
                break

        return self.approximator.value(sim_env.board) + (current_env.score - prev_score)

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        current = node
        discount = 1.0
        while current is not None:
            current.visits += 1
            current.total_reward += reward * discount
            discount *= self.gamma
            current = current.parent

    # def run_simulation(self, root):
    #     node = root
    #     sim_env = self.create_env_from_state(node.state, node.score)
    #
    #     # Selection: Traverse the tree until reaching a non-fully expanded node.
    #     while node.fully_expanded():
    #         action = self.select_child(node)
    #         if action is None:
    #             break
    #         if node.is_max_node:
    #             # Player move (max node)
    #             state, reward, done, _ = sim_env.step(action)
    #             if done:
    #                 break
    #             # Create a chance node after player's move
    #             if action not in node.children:
    #                 node.children[action] = TD_MCTS_Node(sim_env, state, node.score + reward, 
    #                                                      is_max_node=False, parent=node, action=action)
    #             node = node.children[action]
    #         else:
    #             # Chance node (random tile placement)
    #             node = node.children[action]
    #             sim_env = self.create_env_from_state(node.state, node.score)
    #
    #     # Expansion phase
    #     if node.is_max_node and node.untried_actions:
    #         # Expand max node
    #         action = random.choice(node.untried_actions)
    #         node.untried_actions.remove(action)
    #         state, reward, done, _ = sim_env.step(action)
    #         # Create chance node as child
    #         node.children[action] = TD_MCTS_Node(sim_env, state, node.score + reward,
    #                                             is_max_node=False, parent=node, action=action)
    #         node = node.children[action]
    #     elif not node.is_max_node and node.untried_actions:
    #         # Expand chance node
    #         self.expand_chance_node(node, sim_env)
    #         # Select one of the newly created children
    #         if node.children:
    #             action = self.select_child(node)
    #             node = node.children[action]
    #             sim_env = self.create_env_from_state(node.state, node.score)
    #
    #     # Use approximator directly instead of rollout
    #     evaluation = self.approximator.value(node.state)
    #     # Backpropagate the obtained value
    #     self.backpropagate(node, evaluation)
    
    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        while node.fully_expanded():
            action = self.select_child(node)
            if action is None:
                break
            state, _, done, _ = sim_env.step(action)
            if done:
              break
            node = node.children[action]

        # TODO: Expansion: if the node has untried actions, expand one.
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            state, new_score, done, _ = sim_env.step(action)
            node.children[action] = TD_MCTS_Node(sim_env, state, new_score, parent=node, action=action)
            node = node.children[action]

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution

