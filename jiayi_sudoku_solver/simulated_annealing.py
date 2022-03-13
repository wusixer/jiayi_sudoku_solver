"""Module with functions for using simulated annealing algo to solve sudoku"""
import itertools
import random
from collections import defaultdict
from time import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import jiayi_sudoku_solver.Sudoku as Sudoku
from jiayi_sudoku_solver.genetic_algo import *
from jiayi_sudoku_solver.util import *


def decide_which_block_to_not_flip(
    rowwise_correct: List, colwise_correct: List
) -> List:
    """Given rowwise corrected instance and colwise corrected instance, decide 
    which blocks are right and shouldn't be flipped

    Args:
        rowwise_correct (List): a list of length 9 showing the corrected rows, 
                                measured by the number of unique numbers,
                                e.g [8, 7, 8, 7, 7, 6, 9, 8, 8]
        colwise_correct (List): a list of length 9 showing the corrected cols, 
                                measured by the number of unique numbers,
                                e.g [8, 7, 8, 7, 7, 6, 9, 8, 8]

    Returns:
        List: A list of blocks not to be flipped, e.g [0,8]
    """
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


def simulated_annealing(
    startingSudoku: str, max_iteration: int = 500000, temp: float = 10
):
    """Simulated annealing algorithm to solve a sudoku puzzle

    Args:
        startingSudoku (str): the input puzzle represented by a string, e.g 
                                startingSudoku1 =   '
                                                    430260700
                                                    682070493
                                                    107804500
                                                    820190047
                                                    004602910
                                                    950703028
                                                    509306070
                                                    240057106
                                                    703018250
                                                    '
        max_iteration (int): the max number of iteration to try, default is 500000
        temp (float): starting temperature, default is 10

    Returns:
        _type_: _description_
    """
    # get input sudoku
    sudoku_obj = Sudoku.Sudoku(startingSudoku)
    # generate mask for unknown entries for the entire sudoku
    sudoku_mask = mask_for_given_sudoku_val(sudoku_obj.np_Sudoku)
    # generate mask for block entries
    nine_block_mask = get_nine_blocks(sudoku_mask)

    # fill in the initial board
    empty_coord = generate_empty_coord(sudoku_obj.np_Sudoku)
    updated_sudoku = fill_in_np_sudoku(sudoku_obj.np_Sudoku, empty_coord)

    # initial error
    (
        current_best_err,
        current_rowwise_correct,
        current_colwise_correct,
    ) = get_error_from_updated_sukudo(updated_sudoku)
    # print(current_best_err, current_rowwise_correct, current_colwise_correct)

    for iteration in tqdm(range(max_iteration)):
        if current_best_err > 0:
            # ----------------
            # randomly flip two unfilled numbers from a random block
            # identify a block not to flip
            not_to_flip = decide_which_block_to_not_flip(
                current_rowwise_correct, current_colwise_correct
            )
            blocks_to_flip = list(range(0, 9))
            if len(not_to_flip) > 0:
                # remove block from the block list
                blocks_to_flip = [i for i in blocks_to_flip if i not in not_to_flip]
            # randomly pick one block
            block_to_flip = random.choice(blocks_to_flip)

            # flip the two uncertain positions in that block
            flipped_sudoku = randomly_flip_numbers_in_a_block(
                updated_sudoku, nine_block_mask, block_to_flip
            )

            # eval error
            (
                current_error,
                current_rowwise_correct,
                current_colwise_correct,
            ) = get_error_from_updated_sukudo(flipped_sudoku)

            # diff between previous config and current config
            diff = current_error - current_best_err

            # calculate temperature for current epoch
            t = temp / (iteration + 1)

            # calculate metropolis acceptance criterion
            metropolis = np.exp(-diff / t * 10)

            # check if we should keep the new point
            if diff < 0 or metropolis > random.random():
                # print('accept update')
                current_best_err = current_error
                updated_sudoku = flipped_sudoku
                best_solution = updated_sudoku.copy()

        if current_best_err == 0:
            print(f"solved with {iteration} number of iterations")
            return best_solution

    show_sudoku_filled(best_solution)
    print(
        f"max iteration {max_iteration} has reached,  current_best_err is {current_best_err}"
    )
    return best_solution
