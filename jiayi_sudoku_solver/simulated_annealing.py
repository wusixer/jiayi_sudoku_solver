"""Module with functions for using simulated annealing algo to solve sudoku"""
import itertools
import random
from collections import defaultdict
from time import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jiayi_sudoku_solver.util import *


def decide_which_block_to_not_flip(rowwise_correct, colwise_correct):
    not_flip = list()
    # decide which block to not flip
    row_correct_idx = [
        idx for idx, element in enumerate(rowwise_correct) if (element == 9)
    ]
    colwise_correct_idx = [
        idx for idx, element in enumerate(colwise_correct) if (element == 9)
    ]
    if set([0, 1, 2]) <= set(row_correct_idx):
        if set([0, 1, 2]) <= set(colwise_correct_idx):
            not_flip.append(0)
        if set([3, 4, 5]) <= set(colwise_correct_idx):
            not_flip.append(1)
        if set([6, 7, 8]) <= set(colwise_correct_idx):
            not_flip.append(2)

    if set([3, 4, 5]) <= set(row_correct_idx):
        if set([0, 1, 2]) <= set(colwise_correct_idx):
            not_flip.append(3)
        if set([3, 4, 5]) <= set(colwise_correct_idx):
            not_flip.append(4)
        if set([6, 7, 8]) <= set(colwise_correct_idx):
            not_flip.append(5)

    if set([6, 7, 8]) <= set(row_correct_idx):
        if set([0, 1, 2]) <= set(colwise_correct_idx):
            not_flip.append(6)
        if set([3, 4, 5]) <= set(colwise_correct_idx):
            not_flip.append(7)
        if set([6, 7, 8]) <= set(colwise_correct_idx):
            not_flip.append(8)
    return not_flip
