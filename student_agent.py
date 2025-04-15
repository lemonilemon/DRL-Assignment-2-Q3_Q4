# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import copy
import random
import math
from game2048.mcts import MCTSWithExpectimax
from game2048.ntuple import Approximator
from game2048.game import Game2048Env


env = Game2048Env()
approximator = Approximator("game2048/2048.bin")


def get_action(state, score):
    env.board = state.copy()
    env.score = score
    td_mcts = MCTSWithExpectimax(
        env,
        approximator,
        iterations=5,
        exploration_constant=0.1,
        gamma=1,
        norm=130000,
    )
    # Create the root node from the current state
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation()

    best_act, _ = td_mcts.best_action_distribution()
    # legal_actions = [a for a in range(4) if env.is_move_legal(a)]
    # best_act = legal_actions[0]
    # best_value = -math.inf
    # for a in legal_actions:
    #     if td_mcts.root.children[a].value > best_value:
    #         best_value = td_mcts.root.children[a].value
    #         best_act = a
    # best_act = legal_actions[0]
    # best_value = -math.inf
    # for a in legal_actions:
    #     sim_env = copy.deepcopy(env)
    #     new_state, new_score, done, _ = sim_env._step_without_tile(a)
    #     value = approximator.value(new_state)
    #     if value > best_value:
    #         best_value = value
    #         best_act = a
    # print(f"Board state:\n{state}")
    # print("Children nodes:\n")
    # for action, child in td_mcts.root.children.items():
    #     print(
    #         f"action: {action}, child value: {child.value}, child visits: {child.visits}"
    #     )
    print(f"TD-MCTS selected action: {best_act}, current score: {score}")
    # input("Press Enter to continue...")
    return best_act


if __name__ == "__main__":
    env = Game2048Env()
    state = env.reset()
    score = 0
    print(get_action(state, score))
