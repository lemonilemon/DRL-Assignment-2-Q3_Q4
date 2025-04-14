import copy
import random
import math
import numpy as np
import random
from game2048.game import Game2048Env
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Union, Tuple

def create_env_from_state(state, score):
    # Create a deep copy of the environment with the given state and score.
    new_env = Game2048Env()
    new_env.board = state.copy()
    new_env.score = score
    return new_env

# Node types
@dataclass 
class BaseNode:
    probability: float = 1.0
    visits: int = 0
    state: np.ndarray = np.zeros((4, 4), dtype=np.int32)
    parent: Union['Node', None] = None
    score: int = 0
    untried_actions: List = field(default_factory=list)

    def fully_expanded(self):
        return len(self.untried_actions) == 0

@dataclass
class DecisionNode(BaseNode):
    value: float = 0.0  # Total value (Q)
    children: Dict[int, 'ChanceNode'] = field(default_factory=lambda: defaultdict(ChanceNode))
    
    def __post_init__(self):
        env = create_env_from_state(self.state, self.score)
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

@dataclass
class ChanceNode(BaseNode):
    value: float = 0.0 # Computed by the expected value of the children
    probabilities: Dict[Tuple, float] = field(default_factory=lambda: defaultdict(float))
    children: Dict[Tuple, 'DecisionNode'] = field(default_factory=lambda: defaultdict(DecisionNode))
    
    def pull(self, approximator):
        self.value = 0.0
        for tile in self.children.keys():
            if self.children[tile].visits > 0:
                self.value += (self.children[tile].value / self.children[tile].visits) * self.probabilities[tile]
            else:
                self.value += approximator.value(self.children[tile].state) / 100000.0 * self.probabilities[tile]



    def __post_init__(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.state[i][j] == 0]

        count = len(empty_cells)
        for i, j in empty_cells:
            self.untried_actions.append((i, j, 2))
            self.untried_actions.append((i, j, 4))
        for a in self.untried_actions:
            i, j, val = a
            self.probabilities[a] = 0.9 / count if val == 2 else 0.1 / count

Node = Union[DecisionNode, ChanceNode]

class MCTSWithExpectimax:
    def __init__(self, env, approximator, iterations, exploration_constant, gamma):
        self.env = env # Initialized environment
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.gamma = gamma
        self.root = DecisionNode(state=env.board, score=env.score)
        # Add heuristic policy
        def policy(state, actions):
            values = []
            env = create_env_from_state(state, 0)
            for action in actions:
                temp_env = copy.deepcopy(env)
                _, _, _, _ = temp_env._step_without_tile(action)
                value = self.approximator.value(temp_env.board)
                values.append(value)
            if all(v == -float('inf') for v in values):
                return [1.0 / len(actions)] * len(actions)
            max_v = max(values)
            exp_values = [math.exp((v - max_v) / 130000.0) for v in values]
            total = sum(exp_values)
            return [ev / total for ev in exp_values] if total > 0 else [1.0 / len(actions)] * len(actions)
        self.approximator.policy = policy
    
    def select_child(self, node: Node) -> Union[None, int, Tuple]:
        if not node.children:
            return None

        if isinstance(node, DecisionNode):
            if node.visits == 0:
                # If the node has never been visited, return a random child
                return random.choice(list(node.children.keys()))
            actions = list(node.children.keys())
            priors = self.approximator.policy(node.state, actions)
            # Decision node: Use UCT for selection
            uct_values = []
            for action, prior in zip(actions, priors):
                child = node.children[action]
                if child.visits == 0:
                    uct_value = float('inf')
                else:
                    exploitation = child.value
                    exploration = self.c * prior * math.sqrt(math.log(node.visits) / child.visits)
                    uct_value = exploitation + exploration
                uct_values.append((uct_value, action))
            _, best_action = max(uct_values)
            return best_action
        elif isinstance(node, ChanceNode):
            # Chance node: Select a child randomly based on probabilities
            tiles = list(node.children.keys())
            probabilities = [ node.probabilities[tile] for tile in tiles ]
            return random.choices(tiles, weights=probabilities, k=1)[0]

    def rollout(self, sim_env) -> float:
        return self.approximator.value(sim_env.board)


    def backpropagate(self, node, reward):
        reward /= 100000
        current = node
        while current is not None:
            if isinstance(current, DecisionNode):
                # Max node: Update with the reward
                current.value += reward
                current.visits += 1
            elif isinstance(current, ChanceNode):
                # Chance node: Update with expected value (weighted by probability)
                # This is handled implicitly as children propagate their values
                current.pull(self.approximator)
                current.visits += 1
            current = current.parent



    def run_simulation(self):
        node = self.root
        sim_env = copy.deepcopy(self.env)  # Create a new environment for simulation

        # Selection: Traverse until a non-fully expanded node or leaf
        while node.fully_expanded() or isinstance(node, ChanceNode):
            if isinstance(node, ChanceNode) and not node.fully_expanded():
                # Expand chance node
                for tile in node.untried_actions:
                    i, j, val = tile
                    sim_env = create_env_from_state(node.state, node.score)
                    sim_env.board[i][j] = val
                    node.children[tile] = DecisionNode(probability=node.probabilities[tile], state=sim_env.board, score=sim_env.score, parent=node)
                node.untried_actions = []
                # sim_env already updated in expand_chance_node
                sim_env = create_env_from_state(node.state, node.score)
            
            key = self.select_child(node)
            node = node.children[key]

        # Expansion
        if node.untried_actions:
            if isinstance(node, DecisionNode):
                # Expand max node
                actions = node.untried_actions
                priors = self.approximator.policy(node.state, actions)
                action = actions[np.argmax(priors)]
                node.untried_actions.remove(action)
                sim_env = create_env_from_state(node.state, node.score)
                _, _, _, _ = sim_env._step_without_tile(action)
                # Update sim_env to the new state
                node.children[action] = ChanceNode(state=sim_env.board, score=sim_env.score, parent=node)
                node = node.children[action]
            if isinstance(node, ChanceNode):
                # Expand chance node
                for tile in node.untried_actions:
                    i, j, val = tile
                    sim_env = create_env_from_state(node.state, node.score)
                    sim_env.board[i][j] = val
                    node.children[tile] = DecisionNode(probability=node.probabilities[tile], state=sim_env.board, score=sim_env.score, parent=node)
                node.untried_actions = []
                # walk down to the next decision node
                key = self.select_child(node)
                node = node.children[key]


        sim_env = create_env_from_state(node.state, node.score)

        assert isinstance(node, DecisionNode), "Node should be a chance node after expansion"

        # Rollout
        rollout_reward = self.rollout(sim_env)
        # Backpropagate
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self):
        # Compute the normized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in self.root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in self.root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution



    
