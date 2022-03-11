"""Module with functions to manipulate a Sudoku class object"""
import itertools
import random
import time
from collections import defaultdict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jiayi_sudoku_solver.util import *


def randomly_flip_numbers_in_a_block(
    np_Sudoku: np.ndarray, nine_block_masks: List, block_num: int
) -> np.ndarray:
    """Randomly flip two user defined entries within a sudoku block. Note that the block constriants have always 
    been satisfied - meaning each block we have numbers from 1-9 but the order might be wrong.


    Args:
        np_Sudoku (np.ndarray): soduku board that needs to be flipped
        nine_block_masks (List): a mask for the entire board so the position where the solution 
                                needs to be filled is known
        block_num (int): which block to flip

    Returns:
        np.ndarray: an updated sodoku
    """
    updated_sudoku_9_blocks = get_nine_blocks(np_Sudoku)
    one_block_mask = nine_block_masks[block_num]
    row_col_combo_unfilled = return_row_col_combo_unfilled(one_block_mask)
    # print(f"one block mask is {one_block_mask}")
    # print(f"row_col_combo_unfilled is {row_col_combo_unfilled}")
    # randomly pick one position
    swap_position1 = random.choice(row_col_combo_unfilled)
    row_col_combo_unfilled.remove(swap_position1)
    swap_position2 = random.choice(row_col_combo_unfilled)
    tmp = updated_sudoku_9_blocks[block_num][swap_position1[0]][swap_position1[1]]
    updated_sudoku_9_blocks[block_num][swap_position1[0]][
        swap_position1[1]
    ] = updated_sudoku_9_blocks[block_num][swap_position2[0]][swap_position2[1]]
    updated_sudoku_9_blocks[block_num][swap_position2[0]][swap_position2[1]] = tmp
    updated_soduku = convert_nine_blocks_to_sukudo(updated_sudoku_9_blocks)
    return updated_soduku


def return_row_col_combo_unfilled(one_block_mask: np.ndarray) -> List[Tuple]:
    """return a list of (row,col) tuple indicating the location of unfilled positions of the originally provided puzzle
    Args:
        one_block_mask (np.ndarray): one 3x3 block, with 0 indicating the position to fill and 1 means the value in the position is given

    Returns:
        List[Tuple]: a list of (row, col) showing unfilled location of a particular block
    """
    row_col_combo_unfilled = list()
    for row in range(3):
        for col in range(3):
            if one_block_mask[row][col] == 0:
                row_col_combo_unfilled.append((row, col))
    return row_col_combo_unfilled
