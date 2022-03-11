"""Module with functions for using genetic algo to solve sudoku"""
import itertools
import random
from collections import defaultdict
from time import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import jiayi_sudoku_solver.Sudoku as Sudoku
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


def fill_in_np_sudoku(np_Sudoku: np.ndarray, empty_coord: List[Tuple]) -> np.ndarray:
    """Given a np representation of sudoku puzzle and a list of coord with positions that needs to be updated,
    fill the sudoku puzzle up with unique positional solution

    Args:
        np_Sudoku (np.ndarray): A 9X9 np.ndarray representation of the any sudoku
        coord (List[Tuple]): a list of tuple, e.g [(row1,col1), (row2,col2)..] indicating the position
                             that needs to be filled

    Returns:
        np.ndarray: an updated sudoku puzzle by evaluating the input once and fill in the certain number
    """
    # return the number of block updated
    # block_updated = 0
    # pos_val_dict = defaultdict(list, {k: [] for k in coord})
    nine_blocks = get_nine_blocks(np_Sudoku)
    for key in empty_coord:
        # print(key)
        block_num = which_block_it_is(key[0], key[1])
        # print(block_num)
        nums_to_choose = list(range(1, 10))
        # remove block specific numbers
        nums_to_choose = [
            i
            for i in nums_to_choose
            if i not in set(list(nine_blocks[block_num].ravel()))
        ]
        # remove row specific numbers
        # nums_to_choose = [i for i in nums_to_choose if i not in np_Sudoku[key[0]]]
        # remove col specific numbers
        # nums_to_choose = [i for i in nums_to_choose if i not in np_Sudoku.T[key[1]]]
        # num to fill
        num_to_fill = np.random.choice(nums_to_choose)
        np_Sudoku[key[0]][key[1]] = num_to_fill
    return np_Sudoku


def cross_over(for_cross_over: List[np.ndarray]) -> List[np.ndarray]:
    """Given a list of potential sudoku solutions, for each solution1-solution2 pair,
    make new sudoku solution by "crossing over" solution1 and solution2, 
    defined as using the first 5 block configure from solution1 
    and last 4 block configure from solution 2

    Args:
        for_cross_over (List[np.ndarray]): a list of solution to be crossed over

    Returns:
        List[np.ndarray]: cross over result, should be the same length as for_cross_over
    """
    new_cross_over_boards = list()
    new_cross_over_boards.append(
        np.concatenate((for_cross_over[-1][:6], for_cross_over[0][6:]), axis=0)
    )
    for i in range(len(for_cross_over) - 1):
        new_solution = np.concatenate(
            (for_cross_over[i][:6], for_cross_over[i + 1][6:]), axis=0
        )
        new_cross_over_boards.append(new_solution)
    return new_cross_over_boards


def mutation(
    for_mutation: List[np.ndarray], nine_block_mask: List[np.ndarray]
) -> List[np.ndarray]:
    """Given a list of a list of potential sudoku solutions, for each solution, randomly
    flip two entries in a randomly selected block and return new solution

    Args:
        for_mutation (List[np.ndarray]): a list of solution to be crossed over
        nine_block_mask (List[np.ndarray]): output from   `# generate mask for unknown entries for the entire sudoku
                                                            sudoku_mask = mask_for_given_sudoku_val(sudoku_obj.np_Sudoku)
                                                            # generate mask for block entries
                                                            nine_block_mask = get_nine_blocks(sudoku_mask)`

    Returns:
        List[np.ndarray]: mutated solution, should be the same length as `for_mutation`
    """
    new_mutation_boards = list()
    for board in for_mutation:
        block_num = random.choice(range(9))
        new_mutation_boards.append(
            randomly_flip_numbers_in_a_block(board, nine_block_mask, block_num)
        )
    return new_mutation_boards


def genetic_algorithm(
    startingSudoku: str, population_size: int = 100, max_generation: int = 5000
):
    """Genetic algorithm to find a sudoku solution

    Args:
        startingSudoku (str): the input puzzle
        population_size (int, optional): num of offspring for each generation. Defaults to 100.
        max_generation (int, optional): max num of generation to try. Defaults to 5000.
    """
    start = time()
    # get input sudoku
    sudoku_obj = Sudoku.Sudoku(startingSudoku)
    # generate mask for unknown entries for the entire sudoku
    sudoku_mask = mask_for_given_sudoku_val(sudoku_obj.np_Sudoku)
    # generate mask for block entries
    nine_block_mask = get_nine_blocks(sudoku_mask)

    # iteration 0: generate n offsprings unsolved sudoku board and fill in the numbers in each block randomly,
    # defined as fill in entires with unused num in each block for each puzzle
    generation = 0
    boards_to_solve = [sudoku_obj.np_Sudoku.copy() for _ in range(population_size)]
    empty_coord_lst = [generate_empty_coord(board) for board in boards_to_solve]
    updated_boards = [
        fill_in_np_sudoku(board, coord)
        for (board, coord) in zip(boards_to_solve, empty_coord_lst)
    ]

    # for each iteration, do the following
    while generation < max_generation:
        # calculate errors associated with each offspring
        updated_boards_errors = [
            get_error_from_updated_sukudo(board)[0] for board in updated_boards
        ]

        # sort the solution in the order from the best to the worst solution
        updated_boards_ordered = [
            board
            for board in map(
                lambda x: x[1],
                sorted(zip(updated_boards_errors, updated_boards), key=lambda x: x[0]),
            )
        ]

        # calculate the best solution associated error
        min_error = get_error_from_updated_sukudo(updated_boards_ordered[0])[0]

        # if there is no error, this is the solution
        if min_error == 0:
            end = time()
            print(f"Solved in generation {generation}, time used is {end-start}")
            show_sudoku_filled(updated_boards_ordered[0])
            break
        # otherwise, start a new generation
        else:
            new_gen = list()

            # get top 20% elite, elite from previous geneartion will stay to next generation
            elite = updated_boards_ordered[: int(0.2 * population_size)]
            new_gen.extend(elite)

            # get 40% cross over, cross over solutions are randomly chosen and the cross-overed solution will go to next gen
            for_cross_over = random.sample(
                updated_boards_ordered, int(0.4 * population_size)
            )
            new_gen.extend(cross_over(for_cross_over))

            # get 40% mutation, random chosen solution is going to have certain blocks number flipped, will go to the next gen
            for_mutation = random.sample(
                updated_boards_ordered, int(0.4 * population_size)
            )
            new_gen.extend(mutation(for_mutation, nine_block_mask))
            updated_boards = new_gen

            generation += 1

    if generation == max_generation:
        print(f"not solved, min error is {min_error}")
        return updated_boards_ordered[0]
