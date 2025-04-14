import numpy as np
import struct
import math
import copy
from game2048.mcts import create_env_from_state

class Pattern:
    def __init__(self, pattern, iso=8):
        self.pattern = pattern
        self.iso = iso
        self.weights = None
        self.isom = self._create_isomorphic_patterns()

    def _create_isomorphic_patterns(self):
        isom = []
        for i in range(self.iso):
            idx = self._rotate_mirror_pattern([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], i)
            patt = [idx[p] for p in self.pattern]
            isom.append(patt)
        return isom

    def _rotate_mirror_pattern(self, base, rot):
        board = np.array(base, dtype=int).reshape(4,4)
        if rot >= 4:
            board = np.fliplr(board)
        board = np.rot90(board, rot % 4)
        return board.flatten().tolist()

    def load_weights(self, weights):
        self.weights = weights

    def estimate(self, board):
        if self.weights is None:
            raise ValueError("Weights not loaded.")
        total = 0.0
        for iso in self.isom:
            index = self._get_index(iso, board)
            total += self.weights[index]
        return total

    def _get_index(self, pattern, board):
        index = 0
        for i, pos in enumerate(pattern):
            tile = board[pos//4][pos%4]
            if tile == 0: 
                val = 0
            else:
                val = int(np.log2(tile))
            index |= (val & 0xF) << (4 * i)
        return index

class Approximator:
    def __init__(self, bin_path):
        self.patterns = []
        self._load_binary(bin_path)

    def _load_binary(self, path):
        with open(path, 'rb') as f:
            num_features = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_features):
                name_len = struct.unpack('I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                
                # Parse pattern
                pattern = [int(c, 16) for c in name.split()[-1]]
                
                # Create pattern and load weights
                p = Pattern(pattern)
                size = struct.unpack('Q', f.read(8))[0]
                weights = struct.unpack(f'{size}f', f.read(4*size))
                p.load_weights(weights)
                self.patterns.append(p)
    def value(self, board):
        sum = 0
        for pattern in self.patterns:
            sum += pattern.estimate(board)
        return sum
